# Detección de objetos 3D con SECOND en KITTI

Proyecto final de la Electiva de Ingeniería de Software (Universidad de San Buenaventura Cali).
Autor: Juan Pablo Isaziga Benavides. Docente: Carlos Mario Paredes.

Se entrena SECOND (convolución dispersa sobre vóxeles) con la implementación de
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet) en KITTI (partición de Chen et al.:
3 712 escenas de entrenamiento y 3 769 de validación) y se evalúa con las métricas oficiales.

## Contenido

| Carpeta | Qué hay |
|---|---|
| `notebook/SECOND_KITTI.ipynb` | Notebook con el código y todos los resultados (ya ejecutado) |
| `resultados/final/` | Tablas (CSV), registros de evaluación y de entrenamiento, figuras |
| `documento/` | Informe técnico (`informe_final.tex` y PDF) |
| `scripts/` | Análisis exploratorio, figuras BEV, recall por distancia, generador del notebook |
| `scripts/wsl/` | Scripts que se ejecutaron en Ubuntu (WSL2) para entrenar y evaluar |
| `checkpoints/` | Pesos: `second.pth` (preentrenado de OpenPCDet) y el modelo entrenado en este proyecto |

## Entorno usado

NVIDIA GeForce RTX 5060 (8 GB), Ubuntu 22.04 en WSL2, Python 3.9, PyTorch 2.7.1 (CUDA 12.8),
spconv 2.3.8, OpenPCDet 0.6.0 (commit 233f849). Los detalles están en `resultados/final/entorno.txt`.

## Cómo reproducirlo

1. Instalar OpenPCDet y KITTI (con los planos de calzada) como indica su documentación.
2. Entrenar: `bash scripts/wsl/train_second.sh` (unas 7 h en la RTX 5060).
3. Evaluar y generar todas las tablas y figuras: abrir y ejecutar `notebook/SECOND_KITTI.ipynb`.

Ver el notebook no requiere GPU ni instalar nada: ya contiene las salidas.

## Datos

KITTI no se incluye (licencia CC BY-NC-SA 3.0, registro obligatorio):
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>
