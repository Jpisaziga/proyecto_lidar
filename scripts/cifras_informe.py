#!/usr/bin/env python3
"""
Calcula las cifras derivadas que cita el informe (diferencias, caidas, brechas)
a partir de los CSV que genera el notebook en resultados/final/.
Salida: resultados/final/cifras_informe.txt
Uso: python scripts/cifras_informe.py
"""
import csv
import re
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "resultados"
FIN = RES / "final"
CL = ["Car", "Pedestrian", "Cyclist"]
NIV = ["Facil", "Moderado", "Dificil"]


def leer_ap(nombre):
    filas = list(csv.reader(open(FIN / nombre, encoding="utf-8")))
    out = {}
    for f in filas[2:]:
        v = [float(x) for x in f[1:]]
        out[f[0]] = {"3d": v[:3], "bev": v[3:]}
    return out


def r(x):
    return f"{round(x + 1e-9, 2):.2f}"


p, q = leer_ap("ap_r40_propio.csv"), leer_ap("ap_r40_preentrenado.csv")
L = ["Cifras derivadas para el informe (fuente: CSV de resultados/final)"]

L.append("\nPropio - preentrenado, AP R40 (puntos)")
for m in ("3d", "bev"):
    for c in CL:
        L.append(f"{m:<3} {c:<11} " + " ".join(r(a - b) for a, b in zip(p[c][m], q[c][m])))
dif = [a - b for m in ("3d", "bev") for c in CL for a, b in zip(p[c][m], q[c][m])]
L.append(f"rango de diferencias: {r(min(dif))} a {r(max(dif))}")

L.append("\nBrecha automovil - clase, AP_3D R40 moderado")
for nombre, d in (("propio", p), ("preentrenado", q)):
    for c in CL[1:]:
        L.append(f"{nombre:<12} Car - {c}: {r(d['Car']['3d'][1] - d[c]['3d'][1])}")

L.append("\nCaida facil -> dificil, AP_3D R40 (propio)")
for c in CL:
    L.append(f"{c:<11} {r(p[c]['3d'][0] - p[c]['3d'][2])}")
L.append("Diferencia AP_BEV - AP_3D moderado (propio)")
for c in CL:
    L.append(f"{c:<11} {r(p[c]['bev'][1] - p[c]['3d'][1])}")

L.append("\nConvergencia AP_3D moderado: cambio entre evaluaciones")
ep = list(csv.reader(open(FIN / "ap3d_moderado_por_epoca.csv", encoding="utf-8")))
for a, b in zip(ep[1:], ep[2:]):
    L.append(f"epoca {a[0]}->{b[0]}: " + " ".join(f"{c} {r(float(y) - float(x))}"
                                                for c, x, y in zip(CL, a[1:], b[1:])))

L.append("\nRecall por distancia (propio): caida 0-20 m -> 40-70,4 m y diferencia con preentrenado")
rd = list(csv.DictReader(open(FIN / "recall_por_distancia.csv", encoding="utf-8")))
for c in CL:
    f = [x for x in rd if x["clase"] == c]
    L.append(f"{c:<11} caida propio {r(float(f[0]['recall_propio']) - float(f[2]['recall_propio']))}"
             f" | caida preentrenado {r(float(f[0]['recall_preentrenado']) - float(f[2]['recall_preentrenado']))}"
             " | propio - pre por franja: "
             + " ".join(r(float(x['recall_propio']) - float(x['recall_preentrenado'])) for x in f))

L.append("\nPreentrenado: evaluacion final (17-nov) - evaluacion del segundo avance (eval_second.txt)")
txt = (RES / "eval_second.txt").read_text()
for c, iou in (("Car", "0.70"), ("Pedestrian", "0.50"), ("Cyclist", "0.50")):
    i = txt.index(f"{c} AP_R40@{iou}, {iou}, {iou}:")
    b = txt[i:i + 400]
    v3 = [float(x) for x in re.search(r"3d   AP:([\d.]+), ([\d.]+), ([\d.]+)", b).groups()]
    vb = [float(x) for x in re.search(r"bev  AP:([\d.]+), ([\d.]+), ([\d.]+)", b).groups()]
    nuevos = q[c]["3d"] + q[c]["bev"]
    viejos = [round(x + 1e-9, 2) for x in v3 + vb]
    L.append(f"{c:<11} " + " ".join(r(a - b) for a, b in zip(nuevos, viejos)))

pe = list(csv.reader(open(FIN / "perdida_por_epoca.csv", encoding="utf-8")))
L.append(f"\nPerdida media epoca 1: {pe[1][1]} | epoca 80: {pe[-1][1]} | iteraciones totales: {928 * 80}")

(FIN / "cifras_informe.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
