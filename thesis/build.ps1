<#
.SYNOPSIS
    Build the thesis PDF. Works from any directory.

.DESCRIPTION
    LaTeX resolves \input{} paths against the CURRENT WORKING DIRECTORY, not the
    location of the .tex file, so building main.tex from anywhere other than the
    thesis folder fails with "File `frontmatter/titlepage.tex' not found".
    This script cd's to its own location first, so that cannot happen.

    It also calls pdflatex/bibtex directly rather than latexmk, because latexmk is
    a Perl script and Perl is not installed on this machine.

.EXAMPLE
    .\build.ps1              # full build (pdflatex, bibtex, pdflatex x2)
    .\build.ps1 -Quick       # single pass, no bibliography
    .\build.ps1 -Clean       # remove build artefacts
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Clean
)

Set-Location $PSScriptRoot
$main = "main"

# Let MiKTeX fetch any package it is missing without opening a dialog.
$env:MIKTEX_ENABLE_INSTALLER = "t"

if ($Clean) {
    $ext = @("aux","bbl","blg","log","out","toc","lof","lot","fls","fdb_latexmk","synctex.gz")
    # (Remove-ListFiles below covers the subset that a stale-state build needs;
    #  -Clean additionally drops the bibliography and log.)
    foreach ($e in $ext) {
        Remove-Item -LiteralPath "$main.$e" -Force -ErrorAction SilentlyContinue
    }
    Get-ChildItem chapters,frontmatter -Filter "*.aux" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    Write-Output "cleaned"
    return
}

function Invoke-Pdflatex {
    & pdflatex -interaction=nonstopmode "$main.tex" | Out-Null
}

# pdflatex reads main.toc/.lof/.lot from the PREVIOUS run before it rewrites
# them. If those files were produced by a structurally different version of the
# document -- e.g. a report-class ancestor whose \contentsline{chapter} entries
# hit an article-class main.tex, where \l@chapter is undefined and so expands to
# \relax, dropping the table of contents into horizontal mode -- pass 1 fails
# with a cascade of "perhaps a missing \item". The state is self-inflicted and
# recoverable: drop the list files and run the pass again.
function Remove-ListFiles {
    foreach ($e in @("aux","toc","lof","lot","out")) {
        Remove-Item -LiteralPath "$main.$e" -Force -ErrorAction SilentlyContinue
    }
    Get-ChildItem chapters,frontmatter -Filter "*.aux" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

Write-Output "pass 1 ..."
Invoke-Pdflatex

$stale = @(Select-String -Path "$main.log" -Pattern "perhaps a missing" -SimpleMatch -ErrorAction SilentlyContinue)
if ($stale.Count -gt 0) {
    Write-Output "  stale .toc/.aux detected - discarding and repeating pass 1 ..."
    Remove-ListFiles
    Invoke-Pdflatex
}

if (-not $Quick) {
    Write-Output "bibtex ..."
    & bibtex $main | Out-Null
    Write-Output "pass 2 ..."
    Invoke-Pdflatex
    Write-Output "pass 3 ..."
    Invoke-Pdflatex
}

# --- report -------------------------------------------------------------
$errors = @(Select-String -Path "$main.log" -Pattern "^! " -ErrorAction SilentlyContinue)
$undefRef = @(Select-String -Path "$main.log" -Pattern "LaTeX Warning: Reference" -ErrorAction SilentlyContinue)
$undefCite = @(Select-String -Path "$main.log" -Pattern "LaTeX Warning: Citation" -ErrorAction SilentlyContinue)

Write-Output ""
if (Test-Path "$main.pdf") {
    $kb = [math]::Round((Get-Item "$main.pdf").Length / 1KB)
    $pages = (Select-String -Path "$main.log" -Pattern "Output written on $main\.pdf \((\d+) pages" |
              Select-Object -Last 1)
    Write-Output "OK  $main.pdf  ($kb KB)"
    if ($pages) { Write-Output "    $($pages.Matches[0].Groups[1].Value) pages" }
} else {
    Write-Output "FAILED - no PDF produced"
}
Write-Output "    errors               : $($errors.Count)"
Write-Output "    undefined references : $($undefRef.Count)"
Write-Output "    undefined citations  : $($undefCite.Count)"

if ($errors.Count -gt 0) {
    Write-Output ""
    Write-Output "---- first errors ----"
    $errors | Select-Object -First 10 | ForEach-Object { "  " + $_.Line }
}
