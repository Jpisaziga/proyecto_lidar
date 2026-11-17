#!/bin/bash
d=/root/proyecto_lidar/OpenPCDet/output/kitti_models/second/entrenamiento_propio
ls $d 2>/dev/null
log=$(ls -t $d/log_train_*.txt 2>/dev/null | head -1)
[ -n "$log" ] && { wc -l $log; tail -c 2500 $log; }
echo; tail -c 1500 /root/proyecto_lidar/train_second_consola.txt | tr '\r' '\n' | tail -5
echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ls $d/ckpt 2>/dev/null | tail -3
ps aux | grep "train.py" | grep -v grep | wc -l
