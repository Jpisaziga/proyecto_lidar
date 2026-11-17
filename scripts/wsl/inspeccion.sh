#!/bin/bash
source /root/venv/bin/activate
find /root/proyecto_lidar/OpenPCDet/output/kitti_models/second/default -maxdepth 6 | head -20
python -c "import tensorboard; print('tb', tensorboard.__version__)" 2>&1 | tail -1
python -c "import tensorboardX; print('tbx', tensorboardX.__version__)" 2>&1 | tail -1
ls /root/proyecto_lidar/OpenPCDet/tools/*.py
grep -n "add_argument" /root/proyecto_lidar/OpenPCDet/tools/test.py
grep -n "LOGGER_ITER_INTERVAL\|tb_log.add_scalar" -r /root/proyecto_lidar/OpenPCDet/tools/train_utils/train_utils.py | head
