#!/usr/bin/env python3
"""
Exhaustividad (recall) de un detector por clase y franja de distancia en el
conjunto de validacion de KITTI.

Entrada: el result.pkl que deja tools/test.py y kitti_infos_val.pkl.
Se consideran solo los objetos reales que entran en la evaluacion oficial
(nivel facil, moderado o dificil). Un objeto cuenta como detectado si una
prediccion de la misma clase con score >= umbral tiene IoU 3D >= 0,7
(automovil) o 0,5 (peaton y ciclista) con el. El emparejamiento es voraz,
en orden de score, y cada prediccion se usa una sola vez.

La IoU 3D se calcula en CPU (recorte de poligonos en el plano x-y y solape
en altura) para no competir por la GPU.

Uso:
  python analisis_distancia.py --infos ../data/kitti/kitti_infos_val.pkl \
      --result RUTA/result.pkl --umbral 0.3 --salida recall_distancia.csv
"""
import argparse
import csv
import pickle

import numpy as np

CLASES = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}
FRANJAS = [(0, 20), (20, 40), (40, 70.4)]


def nivel_dificultad(altura, oclusion, trunc):
    if altura >= 40 and oclusion == 0 and trunc <= 0.15:
        return 0
    if altura >= 25 and oclusion <= 1 and trunc <= 0.30:
        return 1
    if altura >= 25 and oclusion <= 2 and trunc <= 0.50:
        return 2
    return -1


def esquinas(caja):
    """Esquinas en x-y de una caja (x, y, z, dx, dy, dz, yaw) en orden antihorario."""
    x, y, _, dx, dy, _, yaw = caja[:7]
    local = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2],
                      [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def area(poli):
    if len(poli) < 3:
        return 0.0
    x, y = poli[:, 0], poli[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def recortar(sujeto, clip):
    """Sutherland-Hodgman: interseccion de dos poligonos convexos antihorarios."""
    salida = list(sujeto)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        entrada, salida = salida, []
        if not entrada:
            break

        def dentro(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def cruce(p, q):
            d1 = np.array(q) - np.array(p)
            d2 = b - a
            den = d1[0] * d2[1] - d1[1] * d2[0]
            t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
            return np.array(p) + t * d1

        for j in range(len(entrada)):
            p, q = entrada[j - 1], entrada[j]
            if dentro(q):
                if not dentro(p):
                    salida.append(cruce(p, q))
                salida.append(q)
            elif dentro(p):
                salida.append(cruce(p, q))
    return np.array(salida)


def iou3d(a, b):
    if np.hypot(a[0] - b[0], a[1] - b[1]) > (np.hypot(a[3], a[4]) + np.hypot(b[3], b[4])) / 2:
        return 0.0
    inter_bev = area(recortar(esquinas(a), esquinas(b)))
    zmin = max(a[2] - a[5] / 2, b[2] - b[5] / 2)
    zmax = min(a[2] + a[5] / 2, b[2] + b[5] / 2)
    inter = inter_bev * max(0.0, zmax - zmin)
    union = a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - inter
    return inter / union if union > 0 else 0.0


def franja(d):
    for k, (a, b) in enumerate(FRANJAS):
        if a <= d < b:
            return k
    return -1


def calcular(infos, resultados, umbral):
    """Devuelve {clase: [(detectados, total) por franja]}."""
    por_id = {r["frame_id"]: r for r in resultados}
    cuenta = {c: [[0, 0] for _ in FRANJAS] for c in CLASES}
    for info in infos:
        an = info["annos"]
        n = len(an["gt_boxes_lidar"])
        nombres = an["name"][:n]
        alturas = an["bbox"][:n, 3] - an["bbox"][:n, 1]
        r = por_id[info["point_cloud"]["lidar_idx"]]
        for c, thr in CLASES.items():
            idx = [k for k in range(n) if nombres[k] == c and
                   nivel_dificultad(alturas[k], an["occluded"][k], an["truncated"][k]) >= 0]
            if not idx:
                continue
            gt = an["gt_boxes_lidar"][idx]
            m = (r["name"] == c) & (r["score"] >= umbral)
            pred = r["boxes_lidar"][m][np.argsort(-r["score"][m])]
            usado = np.zeros(len(gt), dtype=bool)
            for p in pred:
                mejor, kmejor = thr, -1
                for k, g in enumerate(gt):
                    if not usado[k]:
                        v = iou3d(p, g)
                        if v >= mejor:
                            mejor, kmejor = v, k
                if kmejor >= 0:
                    usado[kmejor] = True
            for k, g in enumerate(gt):
                f = franja(float(np.hypot(g[0], g[1])))
                if f >= 0:
                    cuenta[c][f][1] += 1
                    cuenta[c][f][0] += int(usado[k])
    return cuenta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--umbral", type=float, default=0.3)
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)
    with open(args.result, "rb") as f:
        resultados = pickle.load(f)
    cuenta = calcular(infos, resultados, args.umbral)

    filas = []
    print(f"Recall por franja (score >= {args.umbral}, IoU 3D 0,7 / 0,5)")
    for c in CLASES:
        linea = f"{c:<11}"
        for (a, b), (det, tot) in zip(FRANJAS, cuenta[c]):
            rec = 100 * det / tot if tot else float("nan")
            linea += f"  {a:>2}-{b:<4} m: {rec:6.2f} % ({det}/{tot})"
            filas.append([c, f"{a}-{b}", det, tot, f"{rec:.2f}"])
        print(linea)
    if args.salida:
        with open(args.salida, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clase", "franja_m", "detectados", "total", "recall_pct"])
            w.writerows(filas)


if __name__ == "__main__":
    main()
