#!/bin/bash
# torch>=2.6 usa weights_only=True por defecto y rechaza los checkpoints que guarda
# OpenPCDet (incluyen escalares de numpy). Los checkpoints son propios: se cargan con weights_only=False.
f=/root/proyecto_lidar/OpenPCDet/pcdet/models/detectors/detector3d_template.py
grep -n "torch.load(" $f
sed -i 's/torch.load(filename, map_location=loc_type)/torch.load(filename, map_location=loc_type, weights_only=False)/' $f
grep -n "torch.load(" $f
grep -rn "torch.load(" /root/proyecto_lidar/OpenPCDet/pcdet /root/proyecto_lidar/OpenPCDet/tools/*.py | grep -v weights_only
