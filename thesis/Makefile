# Build the thesis.  Usage:  make        (full build)
#                            make quick  (single pass, no bibliography)
#                            make clean

MAIN = main

all:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

quick:
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Report anything still unresolved, and every remaining placeholder.
check:
	@grep -c "^! " $(MAIN).log || true
	@grep -c "LaTeX Warning: Citation" $(MAIN).log || true
	@grep -rn "PLACEHOLDER" chapters/ frontmatter/ || echo "no placeholders left"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot chapters/*.aux frontmatter/*.aux

.PHONY: all quick check clean
