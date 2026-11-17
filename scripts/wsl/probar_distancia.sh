#!/bin/bash
source /root/venv/bin/activate
S="/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar/scripts"
cd /root/proyecto_lidar/OpenPCDet/tools
CUDA_VISIBLE_DEVICES="" python - "$S" <<'PY'
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from analisis_distancia import iou3d, esquinas, area, recortar
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
rng = np.random.default_rng(0)
A = np.c_[rng.uniform(0,3,(300,2)), rng.uniform(-1,1,300), rng.uniform(1,4,(300,3)), rng.uniform(-3.2,3.2,300)]
B = A.copy(); B[:, :2] += rng.normal(0,0.8,(300,2)); B[:,6] += rng.normal(0,0.5,300); B[:,3:6] *= rng.uniform(0.7,1.3,(300,3))
ref = boxes_bev_iou_cpu(torch.tensor(A[:, :7]).float(), torch.tensor(B[:, :7]).float()).numpy().diagonal()
mine = []
for a, b in zip(A, B):
    inter = area(recortar(esquinas(a), esquinas(b)))
    mine.append(inter / (a[3]*a[4] + b[3]*b[4] - inter))
mine = np.array(mine)
print("BEV IoU max abs diff vs pcdet:", np.abs(mine - ref).max(), " media ref:", ref.mean())
PY
CUDA_VISIBLE_DEVICES="" python "$S/analisis_distancia.py" --infos ../data/kitti/kitti_infos_val.pkl \
  --result ../output/kitti_models/second/default/eval/epoch_no_number/val/default/result.pkl --umbral 0.3
