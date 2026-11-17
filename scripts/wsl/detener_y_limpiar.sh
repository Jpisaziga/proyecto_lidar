#!/bin/bash
pkill -f "train.py" ; sleep 8
ps aux | grep train.py | grep -v grep | wc -l
rm -rf /root/proyecto_lidar/OpenPCDet/output/kitti_models/second/entrenamiento_propio
nvidia-smi --query-gpu=memory.used --format=csv,noheader
