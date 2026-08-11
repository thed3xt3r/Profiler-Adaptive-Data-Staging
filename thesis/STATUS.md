# Thesis draft - status

Build: `make` (or `pdflatex main` → `bibtex main` → `pdflatex main` ×2).
This machine has no local LaTeX toolchain (no `pdflatex`, no `pandoc`); the
`.tex` sources below have been checked for balanced citations/labels/figure
references but not compiled to PDF from this session. Compile via `build.bat`
/ `build.ps1` (Windows) or wherever the existing `main.pdf` was produced
before trusting page counts/layout.

## Structural conversion to the University of Luxembourg HPC master-thesis template

The document was converted this session from a standalone `book`-class thesis
to the structure required by `thesis/master-thesis-template/` (the university's
own template for the HPC master's programme), using a published example thesis
from the same programme as a concrete model of how a full-length thesis conforms
to it (that example PDF was kept only as a working reference and has since been
removed from the repo). Specifically:

- `\documentclass[11pt,...]{book}` → `\documentclass[10pt,a4paper]{article}`.
  Every `\chapter` became `\section`, every old `\section` became
  `\subsection`, every old `\subsection` became `\subsubsection` (mechanical,
  verified: no bare `\chapter` remains anywhere in `thesis/`). Appendices A
  and B now use `\appendix` + `\section` (lettered) rather than `\chapter`.
- Added the two front-matter pieces the template requires that did not exist
  before: a **Plain Language Summary** (`frontmatter/plainlanguage.tex`,
  jargon-free, general-audience) and an **Abbreviations** table
  (`frontmatter/abbreviations.tex`).
- Rewrote `frontmatter/titlepage.tex` to the template's exact field layout
  (logo, Author / Supervised by / Reviewed by / Examined by, degree line,
  Department of Computer Science, University of Luxembourg). The degree line
  now reads "Master in High Performance Computing" (was previously
  "Information and Computer Sciences" — corrected to match the template and
  the precedent thesis, both specific to the HPC programme).
- Rewrote `frontmatter/declaration.tex` using the university's own Declaration
  of Authorship wording (extracted from the template's included, signed PDF
  example) rather than the previous ad hoc phrasing. No signed PDF exists for
  this thesis, so it remains typeset text with a date/signature line, not
  `\includepdf`.
- Bibliography style changed from `abbrv` to `acm` per the template.
- Copied the university logo
  (`thesis/master-thesis-template/master-thesis-template/univ-luxembourg-logo.eps`)
  into `thesis/figures/` and added `\usepackage{epstopdf}` so `pdflatex` can
  render it; this machine has no `gs`/`epstopdf` to pre-convert and verify the
  logo renders, so **check the title page renders correctly on first compile**.
- All `\ref`/`\label`/`\cite`/`\includegraphics` targets re-verified after the
  conversion (197 refs, 65 labels, 38 cite keys, all resolve; see the
  verification script output from this session if needed).

**Still needs your input**: the Supervised-by / Reviewed-by / Examined-by
mapping on the title page preserves the three names already on file (Dr. Leon
van der Torre, Richard Albrecht, Dr. Andrea Binsfield) but the *mapping* of
those names onto the template's three jury categories is a reasonable guess,
not confirmed — verify the actual jury composition before submission.

## What is written and defensible

| Chapter | State |
|---|---|
| 1 Introduction | Written. Motivation, problem, RQ + 4 sub-questions, contributions. |
| 2 Literature Review | Written. Segmentation architectures, DL for archaeological prospection, I/O for large-scale training. |
| 3 Materials and Datasets | Written. Follows the outline closely; adds measured class-balance and mask statistics. |
| 4 Methodology | Written. Expands the outline: fills the instrumentation TODO, adds explicit classifier thresholds, derives the prefetch-depth formula, adds archive-format rationale and experimental protocol. |
| 5 Results | **Rewritten this session, zero placeholders.** RQ1-RQ4 populated from real cluster runs across all three architectures, plus a new §5.7 documenting the profiler warm-up-bias defect and its fix. |
| 6 Ablation | **Rewritten this session, zero placeholders** for the three sweeps actually run (component, scratch-capacity, threshold sensitivity — 39 jobs, all three architectures). Input-size/CORONA and shuffle-buffer sections were in the original evaluation plan but not run this session; left as explicitly scoped, un-executed future work rather than populated with invented numbers. |
| 7 Conclusions | **Rewritten this session.** RQ answers, headline result, and limitations filled from real data; corrected an earlier draft's inaccurate "loader flags move IoU by five to seven points" claim (real deltas: +3.2, +2.6, −2.5pp — mixed direction, GPU-utilisation impact is the real effect at 24-51pp). |
| 8 Future Work | **Strengthened this session.** "Online rather than one-shot adaptation" now has a concrete evidenced starting point (the profiler-bias diagnostic) instead of being purely speculative. |
| Appendix A Reproduction | **Extended this session** with §A.4, "Closing the gap": the batch-size hypothesis from the original workstation reproduction is now confirmed directly, not just by analogy — a full-protocol replica (`0-reproduction`) run on the cluster at batch 32/20 epochs/10 test passes reached 72.8-73.5% IoU across all three architectures, within 1-1.4 points of the published 74.17±0.38. |
| Appendix B Defects | **Extended this session** with two new accuracy-affecting defect rows (IoU-averaging convention and flip-augmentation semantics, both specific to the main `3-pads/*/` pipeline, not present in the reproduction appendix's own port) and a new §B.4 covering four defects found while collecting this session's evidence: a disk-quota exhaustion bug (scratch dir scoped per-`project_root` instead of shared), a checkpoint-auto-resume-from-truncated-file failure mode, a Lightning 2.x epoch-hook wiring bug in `0-reproduction` itself, and the profiler warm-up-window bias (detailed in Ch.5). |

## Figures generated this session

Ten new figures in `figures/`, all from real data: `ablation-components.png`,
`ablation-scratch-capacity.png`, `ablation-threshold-sensitivity.png`,
`rq3-endtoend.png`, `rq4-prefetch-depth.png`, `profiler-bias-sweep.png`,
`reproduction-gap-closure.png`, and `prediction-examples.png` (real inference
on four held-out test tiles, all three architectures, using the
`0-reproduction` full-protocol checkpoints). It is included in Appendix A
(§"Qualitative output"). Predictions are drawn as thresholded contours rather
than probability heatmaps, with the ground-truth boundary repeated on each
prediction panel; the last row is a deliberately hard (visually complex)
negative tile, on which all three architectures produce a spurious detection.
Generated via `thesis/scripts/make_thesis_figures.py` and
`thesis/scripts/make_prediction_figures.py` (also `inference_bench.py` in the
same directory, used for the throughput numbers cited in Chapter 5) inside the
`tell_seg.sif` container; matplotlib is not available in the bare login-node
Python. Run from the repo root as e.g. `apptainer exec --nv --bind
"$HOME/Thesis:/workspace" tell_seg.sif python thesis/scripts/make_thesis_figures.py`
to regenerate.

## What is still not run

The input-size/CORONA-variant ablation (§6.3) and the shuffle-buffer ablation
(§6.5) were part of the original evaluation plan but were not run this
session — the three-architecture component/scratch/threshold sweeps were
prioritised given available cluster time. Both are written up as scoped,
retained future work rather than populated with placeholder or invented
numbers.

## Before showing anyone

1. **Verify the title page jury mapping** - names are filled in (Dr. Leon van
   der Torre, Richard Albrecht, Dr. Andrea Binsfield) but their assignment to
   the template's Supervised-by / Reviewed-by / Examined-by categories is a
   guess made this session, not confirmed with your actual jury. Also confirm
   the submission month.
2. **Write your own acknowledgements** - currently a placeholder.
3. **Verify every bibliography entry**, including the two added this session
   (`Beseda25`, the steady-state-detection paper; `Coles26`, the Alps/Apertus
   engineering report) — both were fetched from arXiv directly this session
   (author lists and titles confirmed against the abstract pages, arXiv IDs
   2506.04204 and 2604.12973 respectively) but volume/page/DOI fields have not
   been checked against publisher records where applicable. Do not submit a
   bibliography you have not personally confirmed.
4. **Compile and proofread.** This session's edits were checked programmatically
   (every `\cite` key resolves in `references.bib`, every `\ref` resolves to a
   `\label`, every `\includegraphics` file exists) but never actually compiled
   to PDF — do that first, on whichever machine has the LaTeX toolchain, before
   trusting layout, page breaks, or float placement. Pay particular attention
   to the title page: the university logo is an `.eps` file converted at
   compile time via `epstopdf`, which needs Ghostscript on the build machine
   and was never test-compiled this session (this login node has no `gs`).

## Two findings that shaped this session's argument

**The reproduction gap is closed, not just narrowed.** Two concrete bugs in the
main pipeline (IoU-averaging convention, flip-augmentation semantics) explained
only part of the gap between the main pipeline's ~0.35 IoU and the reference's
0.74. The remainder was the workstation's forced batch size 8, now confirmed
directly: a full-protocol cluster replica reaches 72.8-73.5% IoU across all
three architectures. See Appendix A, §A.4.

**The staging mechanism works; the instrument that decides when to use it was
measurably biased.** Shard alone beats loose on GPU utilisation for every
architecture (Ch.6 §6.1); full pre-staging reaches the best GPU utilisation of
any method tested for two of three architectures (Ch.5 §5.3). But the adaptive
profiler selected `loose` in nearly every run across Ch.5 and Ch.6, traced to a
specific, quantified defect: the 8-batch warm-up probe's `T_gpu` measurement was
inflated 6-10× by one-time CUDA/cuDNN warm-up cost baked into too short an
average. A validated fix (discard 5 batches, then measure 8 more) recovers
full-epoch accuracy at roughly the same cost as the original, biased default.
See Ch.5 §5.7 and Appendix B §B.4.
