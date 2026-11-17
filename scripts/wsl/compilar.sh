#!/bin/bash
cd "/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar/documento"
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error informe_final.tex > /tmp/latex.log 2>&1 || { tail -30 /tmp/latex.log; exit 1; }; done
grep -E "Output written|Warning.*(undefined|multiply)|^! " informe_final.log | head -20
