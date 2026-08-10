<#
.SYNOPSIS
    Local Windows/NVIDIA launcher for the PADS DeepLab run.

.DESCRIPTION
    The Windows counterpart to PAD_deeplab.slurm. There is no scheduler and no
    Apptainer container here: the repo-local .venv is invoked directly.

    Defaults are sized for a 4 GB card: batch 4 with 8 accumulation steps gives
    an effective batch of 32, matching the cluster runs.

.EXAMPLE
    .\run_local.ps1
    .\run_local.ps1 -Mode standard -Epochs 20
    .\run_local.ps1 -BatchSize 8 -Accumulate 4      # if you have VRAM to spare
#>
[CmdletBinding()]
param(
    [ValidateSet("standard", "pads", "tune_and_profile")]
    [string]$Mode = "tune_and_profile",
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [int]$Accumulate = 2,
    [string]$Precision = "16-mixed",
    [int]$NumWorkers = 4,
    [int]$TuneTrials = 10,
    [switch]$Fresh,
    [string]$DataRoot = $env:PADS_DATA_ROOT,
    [string]$ProjectRoot = $env:PADS_PROJECT_ROOT
)

$ErrorActionPreference = "Stop"

$ModelDir = $PSScriptRoot
$RepoRoot = Split-Path (Split-Path $ModelDir -Parent) -Parent
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$TrainPy  = Join-Path $ModelDir "train.py"
# Mirror train.py's resolve_project_root (CLI -> env -> repo root) so the tee'd
# log follows -ProjectRoot instead of always landing in the repo, which would
# split a run's log away from its own checkpoints.
if ($ProjectRoot) { $OutRoot = $ProjectRoot } else { $OutRoot = Join-Path $RepoRoot "runs" }
$LogDir   = Join-Path $OutRoot "deeplab\logs"

# ---------- pre-flight ----------
if (-not (Test-Path $Python)) {
    throw "Virtualenv not found: $Python`nCreate it with:`n  python -m venv `"$RepoRoot\.venv`"`n  $RepoRoot\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`n  $RepoRoot\.venv\Scripts\python.exe -m pip install -r `"$RepoRoot\requirements-local.txt`""
}
if (-not (Test-Path $TrainPy)) { throw "Training script not found: $TrainPy" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$gpuName = & $Python -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if ($gpuName -eq "NONE") {
    Write-Warning "torch.cuda.is_available() is False - this run would fall back to CPU."
    Write-Warning "Check that the CUDA build of torch is installed (not the +cpu wheel)."
}

Write-Output "============================================================"
Write-Output "DeepLab Training - DeepLabV3Plus + ResNet-50 + Focal Loss"
Write-Output "Dataset:    bing_1k"
Write-Output "Mode:       $Mode"
Write-Output "GPU:        $gpuName"
Write-Output "Batch:      $BatchSize x $Accumulate accum = $($BatchSize * $Accumulate) effective"
Write-Output "Precision:  $Precision"
Write-Output "Hostname:   $env:COMPUTERNAME"
Write-Output "Start time: $(Get-Date)"
Write-Output "============================================================"

# ---------- build the argument list ----------
$pyArgs = @(
    "-u", $TrainPy,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--accumulate", $Accumulate,
    "--precision", $Precision,
    "--num_workers", $NumWorkers
)
if ($Fresh)       { $pyArgs += "--fresh" }
if ($DataRoot)    { $pyArgs += @("--data_root", $DataRoot) }
if ($ProjectRoot) { $pyArgs += @("--project_root", $ProjectRoot) }

if ($Mode -eq "tune_and_profile") {
    $pyArgs += @("--tune", "--tune_trials", $TuneTrials, "--profile", "--profile_policy")
    $stem = "deeplab_profile"
} elseif ($Mode -eq "pads") {
    # The real PADS path: the profiler measures the pipeline and picks the data
    # policy, training then runs on whatever it picked, and the torch profiler
    # reports on the best checkpoint. No Optuna - tuning is orthogonal and costs
    # days on this card.
    $pyArgs += @("--profile_policy", "--profile")
    $stem = "deeplab_pads"
} else {
    $stem = "deeplab_train"
}

$stamp   = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDir ($stem + "_" + $stamp + ".log")
$ErrFile = Join-Path $LogDir ($stem + "_" + $stamp + ".err")

$env:PYTHONUNBUFFERED = "1"
# Skip albumentations' "new version available" check - a network call on every start.
$env:NO_ALBUMENTATIONS_UPDATE = "1"
# Let the caching allocator grow segments instead of reserving fixed blocks. On a
# 4 GB card a long run can otherwise OOM mid-training purely from fragmentation,
# even though the steady-state footprint fits comfortably.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# stderr goes to its own file rather than being merged into the pipeline: under
# PS 5.1 a merged native stderr is re-wrapped as NativeCommandError records,
# which "Stop" treats as fatal mid-run and which bury real output in noise.
# This also mirrors the SLURM setup's separate %x_%j.out / %x_%j.err.
$ErrorActionPreference = "Continue"
& $Python @pyArgs 2>$ErrFile | Tee-Object -FilePath $LogFile
$exit = $LASTEXITCODE

if ((Test-Path $ErrFile) -and (Get-Item $ErrFile).Length -gt 0) {
    # stderr is redirected to a file, so on failure it would otherwise be
    # invisible at the console. Echo the tail whenever the run did not succeed.
    if ($exit -ne 0) {
        Write-Output ""
        Write-Output "---- last 25 stderr lines ($ErrFile) ----"
        Get-Content $ErrFile -Tail 25
        Write-Output "---- end stderr ----"
    } else {
        Write-Output "stderr (warnings/progress bars) captured in: $ErrFile"
    }
}

Write-Output "============================================================"
Write-Output "Done. Exit code: $exit"
Write-Output "Log: $LogFile"
Write-Output "End time: $(Get-Date)"
Write-Output "============================================================"

exit $exit
