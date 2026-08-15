# Build the thesis.  Usage:  make        (full build)
#                            make quick  (single pass, no bibliography)
#                            make clean

MAIN = main

all:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	@# pdflatex reads main.toc/.lof/.lot from the previous run before rewriting
	@# them; if they came from a structurally different version of the document,
	@# pass 1 dies in a cascade of "perhaps a missing \item". Discard and retry.
	@if grep -q "perhaps a missing" $(MAIN).log; then \
	  echo "  stale .toc/.aux detected - discarding and repeating pass 1 ..."; \
	  rm -f $(MAIN).aux $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).out \
	        chapters/*.aux frontmatter/*.aux; \
	  pdflatex -interaction=nonstopmode $(MAIN).tex; \
	fi
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
