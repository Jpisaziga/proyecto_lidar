#!/usr/bin/env python3
"""
Figura cualitativa: nube de puntos en vista de pajaro con las cajas reales
(verde) y las cajas predichas por SECOND (rojo).

Copiar este archivo y eda_kitti.py a OpenPCDet/tools/ y ejecutar desde ahi
(antes hay que haber corrido create_kitti_infos):

  python bev_predicciones.py \
      --cfg_file cfgs/kitti_models/second.yaml \
      --ckpt RUTA/AL/checkpoint.pth \
      --root ../data/kitti/training \
      --ids 000008 000025 000100 \
      --umbral 0.3

Las escenas se cargan con el mismo KittiDataset (particion val) que usa
test.py, de modo que el modelo recibe exactamente la misma entrada que en la
evaluacion (solo puntos dentro del campo de vision de la camara y del rango
del detector).

Salida: pred_<id>.png por cada escena.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from eda_kitti import leer_etiquetas, leer_calib, centro_en_velodyne, esquinas_bev


def esquinas_pred(caja):
    x, y, _, dx, dy, _, yaw = caja[:7]
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                      [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="carpeta training de KITTI")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--umbral", type=float, default=0.3)
    args = ap.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    root = Path(args.root)

    ds = KittiDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                      training=False, logger=logger)
    pos = {info["point_cloud"]["lidar_idx"]: k for k, info in enumerate(ds.kitti_infos)}

    modelo = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    modelo.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    modelo.cuda().eval()

    rango = cfg.DATA_CONFIG.POINT_CLOUD_RANGE  # [x0, y0, z0, x1, y1, z1]

    for i in args.ids:
        if i not in pos:
            print(f"{i}: no esta en la particion val, se omite")
            continue
        muestra = ds[pos[i]]
        datos = ds.collate_batch([muestra])
        sub = datos["points"][:, 1:4].copy()   # puntos que recibe el modelo
        load_data_to_gpu(datos)
        with torch.no_grad():
            pred, _ = modelo.forward(datos)
        cajas = pred[0]["pred_boxes"].cpu().numpy()
        scores = pred[0]["pred_scores"].cpu().numpy()
        labels = pred[0]["pred_labels"].cpu().numpy()
        keep = scores >= args.umbral
        cajas, scores, labels = cajas[keep], scores[keep], labels[keep]

        velo_a_cam = leer_calib(str(root / "calib" / f"{i}.txt"))
        n_reales = 0
        sigla = {"Pedestrian": "P", "Cyclist": "Ci"}
        ys = [sub[:, 1]]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(sub[:, 0], sub[:, 1], s=0.15, c="0.35", alpha=0.7, linewidths=0)
        otras = False
        print(f"{i}: anotaciones")
        for obj in leer_etiquetas(str(root / "label_2" / f"{i}.txt")):
            if obj["clase"] == "DontCare":
                continue
            cv = centro_en_velodyne(obj, velo_a_cam)
            esq = esquinas_bev(cv, obj["dims"], obj["rot_y"])
            ys.append(esq[:, 1])
            print(f"    {obj['clase']:<14} x={cv[0]:6.1f}  y={cv[1]:6.1f}  "
                  f"dist={np.hypot(cv[0], cv[1]):5.1f}  "
                  f"oclusion={obj['oclusion']}  trunc={obj['truncamiento']:.2f}")
            if obj["clase"] in ("Car", "Pedestrian", "Cyclist"):
                n_reales += 1
                ax.add_patch(Polygon(esq, closed=True, fill=False,
                                     edgecolor="limegreen", linewidth=1.4))
                if obj["clase"] in sigla:
                    ax.text(esq[:, 0].max() + 0.5, esq[:, 1].max(), sigla[obj["clase"]],
                            color="green", fontsize=6)
            else:
                otras = True
                ax.add_patch(Polygon(esq, closed=True, fill=False,
                                     edgecolor="royalblue", linewidth=1.0))
        for c, l in zip(cajas, labels):
            esq = esquinas_pred(c)
            ys.append(esq[:, 1])
            ax.add_patch(Polygon(esq, closed=True, fill=False,
                                 edgecolor="red", linewidth=1.0, linestyle="--"))
            n = cfg.CLASS_NAMES[l - 1]
            if n in sigla:
                ax.text(esq[:, 0].max() + 0.5, esq[:, 1].min() - 1.2, sigla[n],
                        color="red", fontsize=6)
        ax.plot([], [], color="limegreen", label="Real (Car/Ped/Cyc)")
        if otras:
            ax.plot([], [], color="royalblue", label="Real, otra clase")
        ax.plot([], [], color="red", linestyle="--", label="Predicha (SECOND)")
        ax.legend(fontsize=7, loc="upper left")
        y = np.concatenate(ys)
        y0 = max(rango[1], np.floor(y.min()) - 2)
        y1 = min(rango[4], np.ceil(y.max()) + 6)
        ax.set_xlim(rango[0], rango[3])
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Escena {i}: {len(cajas)} detecciones (score >= {args.umbral})",
                     fontsize=9)
        plt.tight_layout()
        plt.savefig(f"pred_{i}.png", dpi=200, bbox_inches="tight")
        plt.close()

        nombres = [cfg.CLASS_NAMES[l - 1] for l in labels]
        print(f"{i}: {n_reales} objetos reales (Car/Ped/Cyc), {len(cajas)} detecciones "
              f"-> pred_{i}.png")
        for c, s, n in zip(cajas, scores, nombres):
            print(f"    {n:<11} score={s:.3f}  x={c[0]:6.1f}  y={c[1]:6.1f}  "
                  f"dist={np.hypot(c[0], c[1]):5.1f}")


if __name__ == "__main__":
    main()
