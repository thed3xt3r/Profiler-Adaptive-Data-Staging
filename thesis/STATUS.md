# Thesis revision status

The revised thesis builds successfully with MiKTeX using:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The resulting `main.pdf` is 61 pages. The final build reports no LaTeX errors,
undefined citations, or undefined references. Representative pages containing
the new dataset overview, mathematical definitions, architecture diagram,
discussion structure, and appendices were rendered to images and visually
checked.

## Revision coverage

- The introduction now formalises semantic segmentation as empirical risk
  minimisation and defines the image, labels, logits, probabilities, and final
  mask.
- Section 2 is named **Related Work** and factual claims were revised and
  supported with primary or authoritative citations, including Bickler.
- The materials chapter now includes a dataset-scope diagram, a compact dataset
  summary table, loss-function definitions, CNN/Transformer explanations, and
  an architecture comparison diagram.
- QGIS, PyTorch, CORONA, and the University of Luxembourg HPC facility are
  cited at their relevant first mentions. The HPC acknowledgement follows the
  facility's official guidance and cites the official ULHPC publication.
- Chapter openings and heading levels were normalised. The former conclusions
  scope subsection was moved into limitations, and Chapter 9 now has a
  dedicated Discussion section with consistently numbered subsections.
- Figure and table references were changed to passive prose where applicable,
  and source-level typo/placeholder/reference checks were performed.

## Items requiring author confirmation

1. Confirm the title-page mapping of the existing names to the Supervisor,
   Reviewer, and Examiner roles, and confirm the submission date.
2. Replace the personal acknowledgements placeholder with the author's own
   text. The required institutional HPC acknowledgement is already present.
3. The input-size/CORONA and shuffle-buffer ablations were not run in the
   underlying research; the thesis explicitly treats them as future work and
   does not invent results.

Some long code identifiers in Appendix B and a few technical lines still
produce modest overfull-box warnings. Visual inspection found them readable;
they do not cause clipping or unresolved content.
