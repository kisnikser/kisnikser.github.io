#!/usr/bin/env bash
# Refresh the site PDFs from the resume submodule.
# 1. Check out the latest main of resume/.
# 2. Build resume.tex and cv.tex with XeLaTeX from TeX Live 2024.
# 3. Copy the PDFs to static/resume.pdf and static/cv.pdf.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

# TeX Live 2024, not the system default (2026). moderncv 2.3.1 is the last
# release whose \makehead still contains the photo patch this CV uses.
tex2024="$HOME/texlive/2024/bin/universal-darwin"
if [[ ! -x "$tex2024/xelatex" ]]; then
  echo "TeX Live 2024 not found at $tex2024" >&2
  exit 1
fi
export PATH="$tex2024:$PATH"
export TEXMFHOME="$HOME/texlive/2024-overleaf/texmf"

git submodule update --init --remote --checkout resume

# Compile next to the .tex files so pictures in assets/ resolve.
(
  cd resume
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error resume.tex
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error cv.tex
)

cp resume/resume.pdf static/resume.pdf
cp resume/cv.pdf static/cv.pdf

find resume -maxdepth 1 \( \
  -name '*.aux' -o -name '*.log' -o -name '*.fls' -o -name '*.fdb_latexmk' \
  -o -name '*.out' -o -name '*.xdv' -o -name '*.synctex.gz' -o -name '*.pdf' \
\) -delete
