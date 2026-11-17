#!/bin/bash
cd /root/descargas_planes
python3 -c "import zipfile;z=zipfile.ZipFile('planes.zip');n=z.namelist();print(len(n),n[:3],n[-2:])"
python3 -c "import zipfile;zipfile.ZipFile('planes.zip').extractall('ext')"
src=$(find ext -type d -name planes | head -1); echo "src=$src"
ls $src | wc -l; head -4 $src/000000.txt
cp -r $src /root/kitti/training/planes
ls /root/kitti/training/planes | wc -l
rm -rf /root/proyecto_lidar/OpenPCDet/output/kitti_models/second/entrenamiento_propio
