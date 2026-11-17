#!/bin/bash
cp /root/proyecto_lidar/OpenPCDet/output/kitti_models/second/entrenamiento_propio/ckpt/checkpoint_epoch_80.pth "/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar/checkpoints/second_propio_epoca80.pth"
sha256sum /root/proyecto_lidar/OpenPCDet/output/kitti_models/second/entrenamiento_propio/ckpt/checkpoint_epoch_80.pth "/mnt/c/Users/JUAN  I/Escritorio/Electiva de ing sw/proyecto_lidar/checkpoints/second_propio_epoca80.pth" | cut -c1-16
