#!/bin/bash
# Uso: ejecutar_notebook.sh [n_celdas]   (sin argumento: notebook completo)
source /root/venv/bin/activate
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
P="/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar"
python "$P/scripts/crear_notebook.py"
cd "$P/notebook"
if [ -n "$1" ]; then
  python - "$1" <<'PY'
import sys, nbformat as nbf
nb = nbf.read("SECOND_KITTI.ipynb", 4); nb.cells = nb.cells[:int(sys.argv[1])]
nbf.write(nb, "_prueba.ipynb")
PY
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 _prueba.ipynb 2>&1 | tail -5
  python - <<'PY'
import nbformat as nbf
nb = nbf.read("_prueba.ipynb", 4)
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain", "")
            if o.get("output_type") == "error": t = "ERROR " + o["ename"] + ": " + o["evalue"]
            print(str(t)[-1500:])
        print("-" * 40)
PY
  rm -f _prueba.ipynb
else
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 SECOND_KITTI.ipynb > ejecucion.log 2>&1
  echo "EXIT: $?" >> ejecucion.log; tail -5 ejecucion.log
fi
