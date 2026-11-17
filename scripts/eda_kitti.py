#!/usr/bin/env python3
"""
Analisis exploratorio del conjunto KITTI 3D Object Detection.

Genera las cifras reales que faltan en la seccion de resultados
preliminares del segundo avance:

  1. Conteo y porcentaje de objetos por clase.
  2. Distribucion de objetos por nivel de dificultad oficial.
  3. Numero de puntos LiDAR dentro de cada caja, por franja de distancia.
  4. Una figura cualitativa: nube de puntos en vista de pajaro con las
     cajas anotadas superpuestas.

Uso:
    python3 eda_kitti.py --root /ruta/a/kitti/training

donde 'training' contiene las carpetas label_2/, velodyne/ y calib/.

Salidas: estadisticas por consola, tablas en LaTeX listas para pegar
(tablas_eda.tex) y figuras en PNG.
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# --- Lectura de etiquetas -------------------------------------------------

def leer_etiquetas(ruta):
    """Devuelve la lista de objetos de un archivo label_2/*.txt."""
    objetos = []
    with open(ruta, "r") as f:
        for linea in f:
            campos = linea.split()
            if len(campos) < 15:
                continue
            objetos.append({
                "clase": campos[0],
                "truncamiento": float(campos[1]),
                "oclusion": int(campos[2]),
                "caja2d": [float(v) for v in campos[4:8]],
                "dims": [float(v) for v in campos[8:11]],      # h, w, l
                "centro": [float(v) for v in campos[11:14]],   # x, y, z en camara
                "rot_y": float(campos[14]),
            })
    return objetos


def nivel_dificultad(obj):
    """Criterio oficial de KITTI: altura de la caja 2D, oclusion y truncamiento."""
    altura = obj["caja2d"][3] - obj["caja2d"][1]
    oc, tr = obj["oclusion"], obj["truncamiento"]
    if altura >= 40 and oc == 0 and tr <= 0.15:
        return "Facil"
    if altura >= 25 and oc <= 1 and tr <= 0.30:
        return "Moderado"
    if altura >= 25 and oc <= 2 and tr <= 0.50:
        return "Dificil"
    return "Descartado"


# --- Calibracion y nube de puntos ----------------------------------------

def leer_calib(ruta):
    datos = {}
    with open(ruta, "r") as f:
        for linea in f:
            if ":" not in linea:
                continue
            clave, valor = linea.split(":", 1)
            datos[clave.strip()] = np.array([float(v) for v in valor.split()])
    tr = np.eye(4)
    tr[:3, :4] = datos["Tr_velo_to_cam"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = datos["R0_rect"].reshape(3, 3)
    return rect @ tr          # velodyne -> camara rectificada


def leer_nube(ruta):
    return np.fromfile(ruta, dtype=np.float32).reshape(-1, 4)


def centro_en_velodyne(obj, velo_a_cam):
    """Pasa el centro del objeto de coordenadas de camara a velodyne."""
    c = np.array([*obj["centro"], 1.0])
    c[1] -= obj["dims"][0] / 2.0          # KITTI da el centro en la base
    return (np.linalg.inv(velo_a_cam) @ c)[:3]


def esquinas_bev(centro_velo, dims, rot_y):
    """Cuatro esquinas de la caja en el plano x-y del velodyne."""
    h, w, l = dims
    yaw = -rot_y - np.pi / 2
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2])
    cos, sin = np.cos(yaw), np.sin(yaw)
    x = centro_velo[0] + dx * cos - dy * sin
    y = centro_velo[1] + dx * sin + dy * cos
    return np.stack([x, y], axis=1)


def puntos_en_caja(nube, centro_velo, dims, rot_y):
    """Cuenta los puntos LiDAR contenidos en la caja 3D.

    centro_velo es el centro geometrico (a media altura, ver
    centro_en_velodyne), asi que la caja va de -h/2 a +h/2 en z.
    """
    h, w, l = dims
    yaw = -rot_y - np.pi / 2
    rel = nube[:, :3] - centro_velo
    cos, sin = np.cos(-yaw), np.sin(-yaw)
    xr = rel[:, 0] * cos - rel[:, 1] * sin
    yr = rel[:, 0] * sin + rel[:, 1] * cos
    dentro = (np.abs(xr) <= l / 2) & (np.abs(yr) <= w / 2) & \
             (np.abs(rel[:, 2]) <= h / 2)
    return int(dentro.sum())


# --- Analisis -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="carpeta 'training' de KITTI")
    ap.add_argument("--muestras-puntos", type=int, default=500,
                    help="cuantas escenas usar para el conteo de puntos")
    ap.add_argument("--escena-figura", default="000008",
                    help="id de la escena para la figura cualitativa")
    args = ap.parse_args()

    dir_lab = os.path.join(args.root, "label_2")
    dir_velo = os.path.join(args.root, "velodyne")
    dir_cal = os.path.join(args.root, "calib")

    ids = sorted(f[:-4] for f in os.listdir(dir_lab) if f.endswith(".txt"))
    print(f"Escenas encontradas: {len(ids)}")

    # 1 y 2: clases y dificultad
    clases = Counter()
    dificultad = defaultdict(Counter)
    for i in ids:
        for obj in leer_etiquetas(os.path.join(dir_lab, f"{i}.txt")):
            if obj["clase"] == "DontCare":
                continue
            clases[obj["clase"]] += 1
            if obj["clase"] in ("Car", "Pedestrian", "Cyclist"):
                dificultad[obj["clase"]][nivel_dificultad(obj)] += 1

    total = sum(clases.values())
    print(f"\nObjetos anotados (sin DontCare): {total}\n")
    print(f"{'Clase':<18}{'Conteo':>10}{'%':>9}")
    for c, n in clases.most_common():
        print(f"{c:<18}{n:>10}{100 * n / total:>8.1f}")

    print(f"\n{'Clase':<14}{'Facil':>9}{'Moderado':>11}{'Dificil':>10}")
    for c in ("Car", "Pedestrian", "Cyclist"):
        d = dificultad[c]
        print(f"{c:<14}{d['Facil']:>9}{d['Moderado']:>11}{d['Dificil']:>10}")

    # 3: puntos por objeto segun distancia
    franjas = [(0, 20), (20, 40), (40, 70)]
    acum = {c: {f: [] for f in franjas} for c in ("Car", "Pedestrian", "Cyclist")}
    for i in ids[:args.muestras_puntos]:
        velo_a_cam = leer_calib(os.path.join(dir_cal, f"{i}.txt"))
        nube = leer_nube(os.path.join(dir_velo, f"{i}.bin"))
        for obj in leer_etiquetas(os.path.join(dir_lab, f"{i}.txt")):
            if obj["clase"] not in acum:
                continue
            cv = centro_en_velodyne(obj, velo_a_cam)
            dist = float(np.hypot(cv[0], cv[1]))
            for f in franjas:
                if f[0] <= dist < f[1]:
                    acum[obj["clase"]][f].append(
                        puntos_en_caja(nube, cv, obj["dims"], obj["rot_y"]))
                    break

    n_esc = min(args.muestras_puntos, len(ids))
    print(f"\nPuntos LiDAR por objeto (mediana), {n_esc} escenas")
    encabezado = f"{'Clase':<14}" + "".join(f"{f'{a}-{b} m':>12}" for a, b in franjas)
    print(encabezado)
    for c in acum:
        fila = f"{c:<14}"
        for f in franjas:
            vals = acum[c][f]
            fila += f"{int(np.median(vals)) if vals else 0:>12}"
        print(fila)
    print("Objetos usados por franja:")
    for c in acum:
        print(f"{c:<14}" + "".join(f"{len(acum[c][f]):>12}" for f in franjas))

    # Figura 1: distribucion de clases
    plt.figure(figsize=(6, 3.2))
    etiquetas = [c for c, _ in clases.most_common()]
    valores = [100 * clases[c] / total for c in etiquetas]
    plt.bar(etiquetas, valores, color="0.45", edgecolor="black", linewidth=0.6)
    for x, v in enumerate(valores):
        plt.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    plt.ylabel("Proporcion (%)")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("distribucion_clases.png", dpi=200)
    plt.close()

    # Figura 2: escena en vista de pajaro con cajas anotadas
    i = args.escena_figura
    velo_a_cam = leer_calib(os.path.join(dir_cal, f"{i}.txt"))
    nube = leer_nube(os.path.join(dir_velo, f"{i}.bin"))
    m = (nube[:, 0] > 0) & (nube[:, 0] < 70.4) & (np.abs(nube[:, 1]) < 40)
    sub = nube[m]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sub[:, 0], sub[:, 1], s=0.08, c=sub[:, 2],
               cmap="viridis", alpha=0.7, linewidths=0)
    colores = {"Car": "red", "Pedestrian": "orange", "Cyclist": "cyan"}
    for obj in leer_etiquetas(os.path.join(dir_lab, f"{i}.txt")):
        if obj["clase"] not in colores:
            continue
        cv = centro_en_velodyne(obj, velo_a_cam)
        esq = esquinas_bev(cv, obj["dims"], obj["rot_y"])
        ax.add_patch(Polygon(esq, closed=True, fill=False,
                             edgecolor=colores[obj["clase"]], linewidth=1.3))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, 70.4)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal")
    ax.set_title(f"Escena {i}: nube de puntos y cajas anotadas (vista de pajaro)",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(f"bev_{i}.png", dpi=200)
    plt.close()

    # Tablas en LaTeX (coma decimal, como el documento)
    def coma(v):
        return f"{v:.1f}".replace(".", ",")

    with open("tablas_eda.tex", "w") as f:
        f.write("% Distribucion de clases\n\\begin{tabular}{@{}lrr@{}}\n\\toprule\n")
        f.write("\\textbf{Clase} & \\textbf{Objetos} & \\textbf{\\%} \\\\\n\\midrule\n")
        for c, n in clases.most_common():
            f.write(f"{c} & {n} & {coma(100 * n / total)} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Total & {total} & 100,0 \\\\\n\\bottomrule\n\\end{{tabular}}\n\n")

        f.write("% Objetos por nivel de dificultad (niveles excluyentes)\n"
                "\\begin{tabular}{@{}lrrrr@{}}\n")
        f.write("\\toprule\n\\textbf{Clase} & \\textbf{Facil} & "
                "\\textbf{Moderado} & \\textbf{Dificil} & \\textbf{Descartado} \\\\\n\\midrule\n")
        for c in ("Car", "Pedestrian", "Cyclist"):
            d = dificultad[c]
            f.write(f"{c} & {d['Facil']} & {d['Moderado']} & {d['Dificil']} "
                    f"& {d['Descartado']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write(f"% Mediana de puntos LiDAR por objeto segun distancia ({n_esc} escenas)\n"
                "\\begin{tabular}{@{}l" + "r" * len(franjas) + "@{}}\n\\toprule\n")
        f.write("\\textbf{Clase} & " +
                " & ".join(f"\\textbf{{{a}--{b} m}}" for a, b in franjas) +
                " \\\\\n\\midrule\n")
        for c in acum:
            celdas = []
            for fr in franjas:
                vals = acum[c][fr]
                celdas.append(f"{int(np.median(vals))} ({len(vals)})" if vals else "--")
            f.write(f"{c} & " + " & ".join(celdas) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("\nArchivos generados: distribucion_clases.png, "
          f"bev_{i}.png, tablas_eda.tex")


if __name__ == "__main__":
    main()
