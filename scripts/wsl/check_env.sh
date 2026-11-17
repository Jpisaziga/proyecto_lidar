#!/bin/bash
source /root/venv/bin/activate
cd /root/proyecto_lidar/OpenPCDet
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
df -h /root | tail -1
ls -la data/kitti data/kitti/ImageSets
ls data/kitti/training
ls data/kitti/*.pkl 2>/dev/null; ls -d data/kitti/gt_database 2>/dev/null
python -c "import torch, spconv, pcdet; print(torch.__version__, torch.cuda.is_available())"
grep -n "NUM_EPOCHS\|BATCH_SIZE\|LR:" tools/cfgs/kitti_models/second.yaml
nproc; free -g | head -2
