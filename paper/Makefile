# SPDX-License-Identifier: CC-BY-4.0
# Build pipeline for the ChronosynD paper, requires latexmk + biber + pdflatex
# (texlive-full on Linux, MiKTeX or TeX Live on Windows). The build is
# hermetic given a TeX distribution; no other external tools are needed.

DOC := paper
LATEXMK := latexmk
LATEXMK_OPTS := -pdf -interaction=nonstopmode -halt-on-error
SOURCES := $(DOC).tex $(wildcard sections/*.tex) references.bib

.PHONY: all clean watch view release

all: $(DOC).pdf

$(DOC).pdf: $(SOURCES)
	$(LATEXMK) $(LATEXMK_OPTS) $(DOC).tex

# Re-run on file change, useful while writing
watch:
	$(LATEXMK) $(LATEXMK_OPTS) -pvc $(DOC).tex

# Open the rendered PDF, platform-specific viewer launchers, override on
# the command line if needed: `make view OPEN=evince`
OPEN ?= xdg-open
view: $(DOC).pdf
	$(OPEN) $(DOC).pdf

# Drop every build artifact, keeps source files
clean:
	$(LATEXMK) -C $(DOC).tex

# Build and copy the PDF to the repo root, the canonical place reviewers
# look for the paper artifact. Run before tagging a release so the root
# PDF reflects the tagged source
release: $(DOC).pdf
	cp build/$(DOC).pdf ../paper.pdf
	@echo "==> rendered PDF copied to ../paper.pdf, ready to commit"
