# Thesis draft — status

Build: `make` (or `pdflatex main` → `bibtex main` → `pdflatex main` ×2).
Currently compiles clean: **47 pages, 0 errors, 0 undefined references, 0 undefined citations.**

## What is written and defensible

| Chapter | State |
|---|---|
| 1 Introduction | Written. Motivation, problem, RQ + 4 sub-questions, contributions. |
| 2 Literature Review | Written. Segmentation architectures, DL for archaeological prospection, I/O for large-scale training. |
| 3 Materials and Datasets | Written. Follows the outline closely; adds measured class-balance and mask statistics. |
| 4 Methodology | Written. Expands the outline: fills the instrumentation TODO, adds explicit classifier thresholds, derives the prefetch-depth formula, adds archive-format rationale and experimental protocol. |
| Appendix A Reproduction | Written, backed by real measurements. |
| Appendix B Defects | Written, backed by real measurements. |

## What is scaffolded, not written

Chapters **5 (Results)**, **6 (Ablation)**, **7 (Conclusions)** and **8 (Future Work)**
have their structure, tables and argument in place, with every numeric claim marked
`\PLACEHOLDER{...}` — these render in **red** in the PDF so they cannot be missed.

Run `make check` to list every remaining placeholder.

Chapters 7 and 8 are otherwise complete prose; only the headline numbers are pending.

## Before showing anyone

1. **Fill the title page** — supervisor / advisor / reviewer names, month.
2. **Write your own acknowledgements** — currently a red placeholder.
3. **Verify every bibliography entry.** The entries in `references.bib` are real
   publications, but volume/page/DOI fields have not been checked against publisher
   records. Do not submit a bibliography you have not personally confirmed.

## Two findings that shape the argument

**Accuracy numbers must come from the cluster.** A faithful reproduction of the
reference configuration on a 4 GB workstation reached IoU 0.4470 ± 0.0059 against a
published 0.7417. The cause is the batch size: the reference trains at 32, which needs
~5 GB and does not fit (measured: 2.44 GB at batch 8, 4.65 GB at 16, OOM at 32).
Batch 8 leaves BatchNorm estimating statistics from 8 samples instead of 32, which
EfficientNet is highly sensitive to. The same code at batch 32 on a V100 reaches
68–73% IoU. Appendix A documents this in full.

**Prior staging results describe decisions, not executions.** The policy was applied
to DataLoader construction but never to dataset construction, so the shard and stage
readers were unreachable code — every run read loose files regardless of what the
profiler selected. Separately, no archive existed in a layout the shard reader could
open. Both are code properties and applied on the cluster too. Appendix B documents
this; the affected experiments need repeating.
