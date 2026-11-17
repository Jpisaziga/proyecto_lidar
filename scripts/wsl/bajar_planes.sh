#!/bin/bash
source /root/venv/bin/activate
grep -n -i "plane" /root/proyecto_lidar/OpenPCDet/docs/GETTING_STARTED.md | head
pip show gdown >/dev/null 2>&1 || pip install -q gdown
mkdir -p /root/descargas_planes && cd /root/descargas_planes
gdown 1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp -O planes.zip
ls -la; file planes.zip
