#!/bin/bash
source /root/venv/bin/activate
pip freeze | grep -i -E "^(torch|numpy|spconv|numba|matplotlib|pandas)=" > /root/freeze_antes.txt
pip install -q ipykernel nbconvert nbformat pandas 2>&1 | tail -3
pip freeze | grep -i -E "^(torch|numpy|spconv|numba|matplotlib|pandas)=" > /root/freeze_despues.txt
diff /root/freeze_antes.txt /root/freeze_despues.txt && echo "SIN CAMBIOS en paquetes clave"
python -m ipykernel install --name venv_lidar --display-name "Python 3.9 (OpenPCDet)" 2>&1 | tail -1
jupyter nbconvert --version
