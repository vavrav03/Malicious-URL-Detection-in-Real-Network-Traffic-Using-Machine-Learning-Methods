set -e

echo "Compiling with XeLaTeX and Biber..."
xelatex -shell-escape ctufit-thesis.tex
biber ctufit-thesis
xelatex -shell-escape ctufit-thesis.tex
xelatex -shell-escape ctufit-thesis.tex

echo "Build complete."