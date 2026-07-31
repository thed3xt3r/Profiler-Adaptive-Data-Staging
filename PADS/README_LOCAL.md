# Running PADS locally on an NVIDIA GPU (Windows)

The three PADS models — `deeplab`, `manet`, `segformer` — were written for the
HPC cluster: SLURM for scheduling, Apptainer for the runtime, and every path
hanging off `~/Thesis`. This document covers the local equivalent. The `.slurm`
files are untouched, so the cluster path still works.

## One-time setup

```powershell
cd "C:\Users\nabee\Source\Thesis local\Profiler-Adaptive-Data-Staging"

python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

The CUDA index URL matters. Plain `pip install torch` pulls the `+cpu` wheel,
which reports `torch.cuda.is_available() == False` and silently trains on the
CPU. Verify:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements-local.txt` floats the versions forward from `requirements.txt`;
the cluster pins (numpy 1.24.3, lightning 2.0.9, …) target Python 3.8 and have
no cp312 wheels.

## Running

Each model directory has a `run_local.ps1` mirroring its `.slurm` file, with the
same two modes:

```powershell
cd PADS\manet
.\run_local.ps1                          # tune_and_profile (default)
.\run_local.ps1 -Mode standard           # plain training run
.\run_local.ps1 -Mode standard -Epochs 5 # quick smoke test
```

Or drive `train.py` directly:

```powershell
..\..\.venv\Scripts\python.exe -u train.py --tune --tune_trials 10 --profile --profile_policy
```

Logs land in `<repo>\runs\<model>\logs\`, checkpoints in
`<repo>\runs\checkpoints_<model>\` (see "Why outputs go to `runs\`" below).
Each run writes two files, mirroring the SLURM `%x_%j.out` / `%x_%j.err` split:
`<model>_<mode>_<stamp>.log` for stdout, `.err` for warnings and progress bars.
Keeping them apart matters on PowerShell 5.1 — merging native stderr into the
pipeline with `2>&1` re-wraps every line as a `NativeCommandError` record, which
buries real output and can abort the run under `$ErrorActionPreference = "Stop"`.

> **Keep the `.ps1` files ASCII.** PowerShell 5.1 reads BOM-less scripts as
> Windows-1252, so a UTF-8 em dash decodes into a sequence containing `U+201D`,
> which the parser treats as a real string delimiter — the script then fails with
> a bogus "string is missing the terminator" error pointing at an unrelated line.
> The same applies to anything `print()`ed from `train.py`, which otherwise shows
> up in your console and logs as `DeepLab â€" DeepLabV3Plus`.

## Memory budget

An RTX 3050 Laptop card has **4 GB** of VRAM. The cluster config — batch 32 at
256×256 in fp32 — does not fit, so training is now `16-mixed` with gradient
accumulation. Every default below keeps the **effective** batch at 32, so loss
curves and IoU stay comparable with the cluster runs.

| Model     | `batch_size` | `accumulate` | Effective | Measured peak |
|-----------|--------------|--------------|-----------|---------------|
| segformer | 16           | 2            | 32        | 1.15 GB (29%) |
| deeplab   | 16           | 2            | 32        | 1.76 GB (44%) |
| manet     | 8            | 4            | 32        | 1.43 GB (36%) |

Measured peak VRAM for one fp16 forward+backward at 256×256, on this card:

| Model     | b4      | b8      | b16     | b32           |
|-----------|---------|---------|---------|---------------|
| segformer | 0.34 GB | –       | 1.15 GB | 2.24 GB       |
| deeplab   | 0.71 GB | 1.06 GB | 1.76 GB | 3.16 GB       |
| manet     | 0.84 GB | 1.43 GB | 2.61 GB | **4.94 GB** ✗ |

The defaults sit near 30–45% to leave headroom for validation, the PyTorch
profiler (which runs with `profile_memory=True`) and allocator fragmentation.

To trade memory for speed, scale the two in opposite directions —
`-BatchSize 32 -Accumulate 1` on segformer keeps the same effective batch and
still fits in 2.24 GB.

> **A crash is not the only failure mode.** MA-Net at batch 32 needs 4.94 GB on
> a 4 GB card and still does *not* raise `CUDA out of memory`: the Windows WDDM
> driver silently spills the excess to system RAM over PCIe. The run completes,
> just far slower than it should. So "it didn't crash" is not evidence a batch
> size fits — check `nvidia-smi` for shared-memory use, or watch for iteration
> rate falling off a cliff.

### fp16 kills MA-Net: use `bf16-mixed` for EfficientNet

**MA-Net defaults to `bf16-mixed`, not `16-mixed`.** With fp16 the training loss
goes NaN around epoch 4 and never recovers:

```
ep  train_loss  val_loss  train_iou  val_iou
 3      0.0384       nan     0.2765   0.3578   <-- last good value
 4         nan       nan     0.2969   0.0000   <-- blowup
 5..18     nan       nan     0.0000   0.0000   <-- dead, 15 wasted epochs
```

The failure is quiet and expensive: every metric collapses to 0, `EarlyStopping`
counts out its full patience on a corpse, and the run reports a "best" score
that is really just the last value before the blowup. `bf16-mixed` on the same
config trains straight through with no non-finite values anywhere:

```
 4      0.0383    0.0414     0.2740   0.3455
 5      0.0378    0.0399     0.2798   0.3539   <-- still improving
```

EfficientNet's SiLU activations and squeeze-excite blocks exceed fp16's narrow
exponent range. bf16 keeps fp32's exponent range (trading mantissa bits, which
matters far less here) and is native on Ampere and later. **DeepLab's ResNet-50
is fp16-safe** — verified over 20 epochs with no NaN — so it keeps `16-mixed`.

A `NonFiniteLossGuard` callback now aborts the run at the epoch the loss goes
non-finite, naming `bf16-mixed` in the error. It checks the epoch *mean*, not
individual batches: single batches may legitimately hit inf under AMP and
`GradScaler` handles those, so a per-batch check would false-positive.

> If you change encoders, re-check this. The rule of thumb: ResNet/VGG-style
> encoders tolerate fp16; EfficientNet and other SiLU/SE-heavy backbones often
> do not.

### Is `16-mixed` safe with focal loss?

Yes — checked directly rather than assumed, because MA-Net once reported
`valid/iou = 0.0000` where SegFormer (which uses `BCEWithLogitsLoss`) trained
fine, and MA-Net/DeepLab are the two that use `smp.losses.FocalLoss`.

Comparing one forward+backward in fp32 against the same under
`autocast(float16)`, on a ~3%-positive target:

```
focal  loss fp32=1.518682 fp16=1.518682  grad_rel_err=0.000e+00  finite=True
bce    loss fp32=1.750406 fp16=1.750406  grad_rel_err=0.000e+00  finite=True
```

Gradients are bit-identical: autocast keeps `logsigmoid` in fp32, so there is no
underflow. The zero IoU was the learning rate — that trial had Optuna's sampled
`lr=0.00107`, which collapses the model to all-background within one epoch on
this imbalanced task. At the default `1e-4`, one epoch gives IoU ≈ 0.037.

Worth knowing for the writeup: **this task converges slowly.** A handful of
epochs is a pipeline shakeout, not a result. The cluster config runs to 100
epochs with `patience=15` for a reason.

## Accuracy: why the variants scored far below the paper

The MA-Net baseline is from *"A human–AI collaboration workflow for
archaeological sites detection"*, which reports **~74% IoU** on `bing_1k`. Local
runs were reaching ~33%. `baseline/` and `PADS/manet` are byte-identical for the
loose path, so this was never a PADS regression — both had drifted from the
repo-root reference implementation (`dataset.py` + `model.py` at the top level).

Five differences, in rough order of impact:

| # | Aspect | Root reference | baseline / PADS (before) | Status |
|---|--------|----------------|---------------------------|--------|
| 1 | Input scaling | `A.Normalize` → [0,1] then std | none; model standardised **raw 0–255** | **fixed** |
| 2 | Resolution | 512×512 | 512 crop → `Resize(256)` | **fixed** |
| 3 | Focal loss α | `alpha=0.75` (class-weighted) | `alpha=None` (unweighted) | **fixed** |
| 4 | Val crop | `CenterCrop` (deterministic) | `RandomCrop` (random each epoch) | **fixed** |
| 5 | Augmentation | p=0.5 + ShiftScaleRotate/ColorJitter/GaussNoise | p=0.25, brightness/contrast only | left as-is |

### 1. The input scale bug (the big one)

`ToTensorV2` only reorders HWC→CHW. Unlike torchvision's `ToTensor` it does
**not** divide by 255 — and the variants' transforms have no `A.Normalize`. The
models then applied smp's ImageNet statistics, which assume `input_range=[0,1]`:

```
[dataset] image dtype=torch.uint8  min=27.0  max=255.0
[model]   AS-IS   normalised = [118.2, 1136.4]  mean=672.4     <-- ~500x out of range
[model]   IF /255 normalised = [ -1.64,   2.64]  mean=0.66     <-- correct
```

The pretrained encoder was being fed values ~500× larger than it was trained on,
which makes the ImageNet weights near-useless. Fixed in each `model.py::forward`
with `image = image.float() / 255.0` before standardising.

> **SegFormer was already correct** and is deliberately *not* changed:
> `segformer/dataset.py::_finalize()` does `image.astype(np.float32) / 255.0`
> itself. Adding the same rescale there would divide twice.

### 2–4. Resolution, loss weighting, validation crop

- Source images are **1024×1024**. The pipeline crops 512×512, then used to
  `Resize(256)` — discarding 3/4 of the pixels, which matters for small sites.
  The resize is gone; training is at 512.
- Only ~12% of pixels are positive (median 7%), but smp's `FocalLoss` defaults to
  `alpha=None`, i.e. no class weighting, so the model could coast on predicting
  background. Now `alpha=0.75` (`focal_alpha`/`focal_gamma` in config).
- Validation used `RandomCrop`, scoring a *different* random patch of each image
  every epoch, so val IoU jittered for reasons unrelated to the model. Now
  `CenterCrop`.

### Effect

Normalization alone, on DeepLab (3 epochs, everything else unchanged):

| Epoch | Before | After |
|-------|--------|-------|
| 1 | 0.2428 | 0.2422 |
| 2 | — | **0.3138** |
| 11 | 0.3287 (best of 20) | — |

It reached in 2 epochs what previously took ~9–11 — roughly 5× faster
convergence. Resolution and loss weighting are expected to add more.

### VRAM after moving to 512

512×512 is 4× the pixels, so the batch defaults changed. Measured peak, fp16:

| Model | b2 | b4 | b8 | b16 | default |
|-------|------|------|------|------|---------|
| manet | 1.41 | **2.56** | 4.88 ✗ | — | 4 × 8 |
| deeplab | — | **1.76** | 3.16 | — | 4 × 8 |
| segformer | — | — | **2.24** | 4.41 ✗ | 8 × 4 |

✗ exceeds the 4 GB card and only "works" via WDDM system-RAM spill. Effective
batch stays 32 everywhere.

## Watching a run

Three live views, in increasing order of usefulness:

```powershell
# 1. stdout, streamed (the PowerShell equivalent of tail -f)
Get-Content "<repo>\runs\deeplab\logs\deeplab_train_<stamp>.log" -Wait -Tail 30

# 2. TensorBoard - loss/IoU curves updating as it trains
& "<repo>\.venv\Scripts\tensorboard.exe" --logdir "<repo>\runs\checkpoints_deeplab"
#   then open http://localhost:6006

# 3. training_curves.png in runs\checkpoints_<model>\ - written at end of training
```

`nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` is the quick
check that work is actually on the GPU.

### Resuming, and `--fresh`

All three models resume from `last.ckpt` in their checkpoint directory if one is
there, rather than silently restarting from epoch 0. This came from the cluster
(jobs on `--qos=besteffort` get preempted) and is just as useful locally after a
crash or reboot. Originally only `segformer` did this; `manet` and `deeplab` now
match.

The consequence to know about: **re-running a finished config appears to do
nothing.**

```
[Resume] Found existing checkpoint at ...\checkpoints_deeplab\last.ckpt; resuming training from there.
`Trainer.fit` stopped: `max_epochs=20` reached.
```

That is correct behaviour — the checkpoint had already reached the epoch cap.
To start over instead, pass `--fresh` (or `-Fresh` to `run_local.ps1`):

```powershell
.\run_local.ps1 -Mode standard -Epochs 20 -Fresh
```

Asking for *fewer* epochs than the checkpoint already ran is a hard error in
Lightning, so it is reported with the two ways out rather than a stack trace:

```
[Resume] You restored a checkpoint with current_epoch=10, but you have set Trainer(max_epochs=1).
         ...\checkpoints_deeplab\last.ckpt
         has already trained past --epochs 1. Either raise
         --epochs to continue it, or pass --fresh (-Fresh) to start over.
```

> `-Fresh` writes into the *existing* checkpoint directory: Lightning adds
> `last-v1.ckpt` alongside `last.ckpt` and **overwrites `training_curves.png`**.
> It does not delete the previous best checkpoint, and the full metric history
> survives in `lightning_logs\version_N`, so a clobbered curve can be rebuilt
> from the event file. Even so, point `-ProjectRoot` somewhere disposable when
> you are only testing.

### Checkpoint filenames and the `valid/iou` slash

The monitored metric is `valid/iou`. With Lightning's default
`auto_insert_metric_name=True`, that name is interpolated into the filename
*verbatim* — and on Windows the slash is a **path separator**, so Lightning
silently creates a directory:

```
checkpoints_deeplab\deeplab-resnet50-trial0-epoch=10-valid\iou=0.3287.ckpt
                                                         ^ a directory
```

`save_top_k=1` still prunes correctly, but it deletes the `.ckpt` files and
leaves the empty parent directories behind, so they accumulate one per improved
epoch. Harmless on the cluster's Linux filesystem, which is why it went
unnoticed. All three models now pass `auto_insert_metric_name=False` and spell
the labels out, yielding a flat `...-epoch10-iou0.3287.ckpt`.

## Paths

Nothing is hardcoded to `~/Thesis` any more. Both roots resolve from CLI flag →
environment variable → filesystem probe:

| Variable             | Meaning                                          | Default                    |
|----------------------|--------------------------------------------------|----------------------------|
| `PADS_PROJECT_ROOT`  | Where checkpoints, logs and scratch are written   | `<repo>\runs`              |
| `PADS_DATA_ROOT`     | Directory *containing* the `bing_1k` folder       | probed (see below)         |
| `PADS_SCRATCH_DIR`   | Staging area for the PADS `stage` policy          | `<project_root>\scratch`   |
| `PADS_SHARD_DIR`     | Tar archives for the PADS `shard` policy          | `<data_root>\bing_1k_tars` |

### Why outputs go to `runs\`, not the repo root

The cluster convention puts outputs one level *above* the code —
`~/Thesis/PADS/deeplab/train.py` writes to `~/Thesis/deeplab/logs/` and
`~/Thesis/checkpoints_deeplab/`. Applied literally here, that drops PADS logs
straight inside the repo's **baseline** `deeplab/` and `segformer/` source
directories, leaving no way to tell which variant produced a given log.

So `PROJECT_ROOT` defaults to `<repo>\runs` instead:

```
runs\
  deeplab\logs\          segformer\logs\        manet\logs\
  checkpoints_deeplab\   checkpoints_segformer\ checkpoints_manet\
  scratch\
```

Everything *below* the root keeps the cluster's structure, so
`PADS_PROJECT_ROOT=~/Thesis` reproduces the original layout exactly — which is
what the `.slurm` jobs still expect. `runs/` is gitignored.

The data-root probe tries `<repo>\data`, then `..\Thesis\source`, then
`~/Thesis`, and picks the first that actually contains `bing_1k\train` — which
is why the same file still works unmodified on the cluster. Override per run:

```powershell
.\run_local.ps1 -DataRoot "D:\datasets" -ProjectRoot "D:\pads-runs"
```

If the dataset can't be found the run now fails immediately with the path it
tried, rather than dying later inside `os.listdir`.

## What changed in `train.py`

Beyond paths and batch sizing:

- **Removed the `_is_ampere_or_later` monkey-patch.** It forced Lightning to
  treat the GPU as pre-Ampere to dodge a cluster driver bug. On a local RTX 30xx
  it only disables the tensor-core paths mixed precision wants.
- **`deterministic="warn"` instead of `True`.** Several cuDNN kernels here have
  no deterministic implementation; strict mode turns those into hard errors.
- **Optuna pruning callback is now optional.** optuna ≥ 4 moved the Lightning
  integration to the separate `optuna-integration` package, and it imports
  `pytorch_lightning` rather than `lightning`. A failure there now degrades to
  "no pruning" instead of killing the study. (`segformer` already did this;
  `manet` and `deeplab` now match.)
- **Optuna studies persist to SQLite** at
  `checkpoints_<model>\optuna_study.db`, and completed trials are counted on
  startup. An OOM or Ctrl-C part-way through a 10-trial sweep now resumes
  instead of restarting from trial 0. (Again, `segformer` had this already.)
- **The scratch directory is created before use.** `StagingArcheoDataset`
  swallows copy errors, so a missing directory quietly downgraded the `stage`
  policy to an ordinary read — the policy looked selected but did nothing.
- **Shard tar paths are wired up.** `create_dataset` accepted
  `originals_tar_path` / `negs_tar_path` but `main()` never passed them, so the
  `shard` policy always fell back to `loose`. They now resolve from
  `PADS_SHARD_DIR` when the archives exist.

## A caveat on the `stage` policy

`select_policy()` returns `stage` when host-to-device time dominates, on the
assumption that the dataset lives on a slow shared filesystem and node-local
scratch is genuinely faster. Locally both source and scratch are the same SSD,
so staging is pure copy overhead. The machinery is left intact — the policy
selection is the thesis contribution — but treat local `stage` timings as
non-representative of the cluster result.
