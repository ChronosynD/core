# SPDX-License-Identifier: CC-BY-4.0
# Per-project latexmk configuration, picked up automatically when latexmk runs in this directory

$bibtex_use = 2;

$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

$out_dir = 'build';

$bibtex_silent_switch = '-q';

push @generated_exts, 'synctex.gz', 'bbl', 'run.xml';
