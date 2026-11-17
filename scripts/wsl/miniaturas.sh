#!/bin/bash
which pdftoppm || apt-get install -y -qq poppler-utils >/dev/null 2>&1; which pdftoppm
mkdir -p /tmp/pags && rm -f /tmp/pags/*
pdftoppm -r 45 -png "/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar/documento/informe_final.pdf" /tmp/pags/p
cd /tmp/pags && /root/venv/bin/python -c "
from PIL import Image; import glob
fs=sorted(glob.glob(\"p-*.png\")); ims=[Image.open(f) for f in fs]; w,h=ims[0].size
out=Image.new(\"RGB\",(w*7,h*3),\"white\")
for k,im in enumerate(ims): out.paste(im,((k%7)*w,(k//7)*h))
out.save(\"/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar/scripts/wsl/_paginas.png\")
print(len(fs))"
