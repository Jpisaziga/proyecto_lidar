# Guía para obtener los resultados preliminares (SECOND en KITTI)

Recomendado: Ubuntu 20.04/22.04 o WSL2 con Ubuntu. OpenPCDet en Windows nativo es muy problemático.
Necesitas: driver NVIDIA, Python 3.8–3.10, ~60 GB libres.

## 0. Revisar la GPU
```bash
nvidia-smi
```
Anota el modelo de la GPU y la versión de CUDA que muestra (van en el informe).

## 1. Entorno
```bash
conda create -n pcdet python=3.9 -y
conda activate pcdet
# PyTorch: usa el comando de https://pytorch.org/get-started/locally/ según tu CUDA
pip install spconv-cu118        # cambia 118 por tu CUDA (113, 114, 116, 117, 120...)
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
```
Si algo falla aquí, mándame el error completo.

## 2. Datos KITTI (requiere cuenta gratuita en cvlibs.net)
Descargar de https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d :
- Velodyne point clouds (29 GB)
- Training labels (5 MB)
- Camera calibration matrices (16 MB)
- Left color images (12 GB)  <- se necesita para generar los índices

Estructura final:
```
OpenPCDet/data/kitti/
  ImageSets/{train.txt, val.txt, test.txt}    (vienen en el repo; si no, ver nota)
  training/{calib, velodyne, label_2, image_2}
  testing/{calib, velodyne, image_2}
```
Nota: train.txt y val.txt son la partición 3712/3769 de Chen et al. Verifica con
`wc -l data/kitti/ImageSets/*.txt` (debe dar 3712 y 3769).

Generar los índices (desde la raíz de OpenPCDet, tarda unos minutos):
```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## 3. Checkpoint preentrenado de SECOND
En el README de OpenPCDet, tabla "KITTI 3D Object Detection Baselines", fila SECOND:
descarga el .pth del enlace de esa fila. Guárdalo en `OpenPCDet/checkpoints/second.pth`.
(Si prefieres entrenar tú mismo ver el paso 6.)

## 4. Evaluación -> tabla de AP (resultado principal)
```bash
cd tools
python test.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --ckpt ../checkpoints/second.pth
```
Al final imprime y guarda en `output/kitti_models/second/default/eval/.../log_eval_*.txt`
bloques como "Car AP@0.70, 0.70, 0.70 ... 3d AP: ..." con R11 y R40.
**Mándame ese bloque completo (copiar y pegar).**
Usamos R40, que es lo que dice el documento.

## 5. Análisis exploratorio y figuras
```bash
cp /RUTA/eda_kitti.py /RUTA/bev_predicciones.py .       # a OpenPCDet/tools
python eda_kitti.py --root ../data/kitti/training
python bev_predicciones.py --cfg_file cfgs/kitti_models/second.yaml \
    --ckpt ../checkpoints/second.pth --root ../data/kitti/training \
    --ids 000008 000025 000100 --umbral 0.3
```
Salidas: tablas_eda.tex, distribucion_clases.png, bev_*.png, pred_*.png.
Mándame: lo que imprime eda_kitti.py en consola y los pred_*.png.

## 6. (Opcional, más fuerte para la nota) entrenar tú mismo
```bash
python train.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --epochs 80
```
Tarda horas (depende de la GPU). Si te quedas sin memoria, baja a `--batch_size 2`.
Luego evalúas con test.py usando el checkpoint de `output/.../ckpt/`.
Si solo haces el paso 4, el documento debe decir que los números vienen de pesos preentrenados.

## Qué me tienes que mandar
1. Modelo de GPU y CUDA.
2. El bloque de AP de test.py.
3. La salida en consola de eda_kitti.py.
4. Los pred_*.png (los que se vean mejor, 2 o 3).
5. Si usaste pesos preentrenados o entrenaste tú.
