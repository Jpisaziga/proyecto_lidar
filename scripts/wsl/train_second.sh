#!/bin/bash
# Entrenamiento propio de SECOND en KITTI (train 3712) y evaluacion final en val (3769).
# Si se interrumpe, volver a ejecutarlo: train.py reanuda desde el ultimo checkpoint guardado.
source /root/venv/bin/activate
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
cd /root/proyecto_lidar/OpenPCDet/tools
python train.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --epochs 80 \
  --extra_tag entrenamiento_propio --max_ckpt_save_num 80 --fix_random_seed \
  > /root/proyecto_lidar/train_second_consola.txt 2>&1
echo "EXIT CODE: $?" >> /root/proyecto_lidar/train_second_consola.txt
