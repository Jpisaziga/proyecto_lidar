#!/bin/bash
d=/root/proyecto_lidar/OpenPCDet/output/kitti_models/second/entrenamiento_propio
echo "checkpoints: $(ls $d/ckpt | grep -c checkpoint_epoch)"
ls -la $d/ckpt/checkpoint_epoch_80.pth
grep 'End training' $d/train_*.log | cut -c1-120
grep 'Train:   80/80' $d/train_*.log | tail -1 | cut -c1-110
