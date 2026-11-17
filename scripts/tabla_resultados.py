#!/usr/bin/env python3
"""
Lee el log de tools/test.py (OpenPCDet, KITTI val) y genera:
  - tabla_ap.tex: AP_3D y AP_BEV con 40 umbrales (R40), coma decimal.
  - comparacion_ap.txt: las cifras redondeadas a dos decimales y las
    diferencias que se citan en el documento.

Uso: python tabla_resultados.py resultados/eval_second.txt resultados/
"""
import re
import sys
from pathlib import Path

CLASES = [("Car", "0.70", "Automóvil", "0,70"),
          ("Pedestrian", "0.50", "Peatón", "0,50"),
          ("Cyclist", "0.50", "Ciclista", "0,50")]

# Tabla tab:baseline del documento (yan2018, val, automovil, 11 umbrales)
PAPER_R11 = {"3d": [87.43, 76.48, 69.10], "bev": [89.96, 87.07, 79.66]}
# Tabla tab:openpcdet (README OpenPCDet, val, moderado, 11 umbrales)
README_R11_MOD = {"Car": 78.62, "Pedestrian": 52.98, "Cyclist": 67.15}


def leer(log):
    texto = Path(log).read_text()
    res = {}
    for clase, iou, _, _ in CLASES:
        for var in ("AP", "AP_R40"):
            cab = f"{clase} {var}@{iou}, {iou}, {iou}:"
            i = texto.index(cab)
            bloque = texto[i:i + 400].splitlines()[1:5]
            for linea in bloque:
                m = re.match(r"(bbox|bev|3d|aos)\s+AP:([\d.]+), ([\d.]+), ([\d.]+)", linea)
                if m:
                    res[(clase, var, m.group(1))] = [float(m.group(k)) for k in (2, 3, 4)]
    return res


def r2(v):
    return round(v + 1e-9, 2)


def coma(v):
    return f"{v:.2f}".replace(".", ",")


def main():
    log, salida = sys.argv[1], Path(sys.argv[2])
    res = leer(log)

    # Tabla LaTeX R40: filas = clase, columnas = 3D (F/M/D) | BEV (F/M/D)
    lineas = ["\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
              " & & \\multicolumn{3}{c}{\\textbf{AP$_{3D}$}} & "
              "\\multicolumn{3}{c}{\\textbf{AP$_{\\text{BEV}}$}} \\\\",
              "\\cmidrule(lr){3-5} \\cmidrule(l){6-8}",
              "\\textbf{Clase} & \\textbf{IoU} & \\textbf{Fácil} & \\textbf{Moderado} & "
              "\\textbf{Difícil} & \\textbf{Fácil} & \\textbf{Moderado} & \\textbf{Difícil} \\\\",
              "\\midrule"]
    for clase, _, nombre, iou in CLASES:
        v3 = [coma(r2(x)) for x in res[(clase, "AP_R40", "3d")]]
        vb = [coma(r2(x)) for x in res[(clase, "AP_R40", "bev")]]
        lineas.append(f"{nombre} & {iou} & " + " & ".join(v3 + vb) + " \\\\")
    lineas += ["\\bottomrule", "\\end{tabular}"]
    (salida / "tabla_ap.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    out = []
    out.append("Cifras de resultados/eval_second.txt redondeadas a 2 decimales")
    for var in ("AP_R40", "AP"):
        etiqueta = "R40" if var == "AP_R40" else "R11"
        for met in ("3d", "bev"):
            for clase, _, _, _ in CLASES:
                v = [r2(x) for x in res[(clase, var, met)]]
                out.append(f"{etiqueta} {met:<3} {clase:<11} F/M/D = {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}")

    out.append("\nBrecha entre clases, AP_3D R40 moderado (calculada con los valores de 2 decimales)")
    car = r2(res[("Car", "AP_R40", "3d")][1])
    for c in ("Pedestrian", "Cyclist"):
        v = r2(res[(c, "AP_R40", "3d")][1])
        out.append(f"Car - {c}: {car:.2f} - {v:.2f} = {r2(car - v):.2f}")

    out.append("\nComparacion con tab:baseline (yan2018, automovil, R11): propio R11 - paper")
    for met in ("3d", "bev"):
        propio = [r2(x) for x in res[("Car", "AP", met)]]
        dif = [r2(p - q) for p, q in zip(propio, PAPER_R11[met])]
        out.append(f"{met:<3} propio = {propio}  paper = {PAPER_R11[met]}  dif = {dif}")

    out.append("\nComparacion con tab:openpcdet (README, R11 moderado): propio - README")
    for c in ("Car", "Pedestrian", "Cyclist"):
        p = r2(res[(c, "AP", "3d")][1])
        out.append(f"{c:<11} propio = {p:.2f}  README = {README_R11_MOD[c]:.2f}  dif = {r2(p - README_R11_MOD[c]):.2f}")

    out.append("\nDiferencia R40 - R11, AP_3D moderado")
    for c in ("Car", "Pedestrian", "Cyclist"):
        a = r2(res[(c, "AP_R40", "3d")][1])
        b = r2(res[(c, "AP", "3d")][1])
        out.append(f"{c:<11} R40 = {a:.2f}  R11 = {b:.2f}  dif = {r2(a - b):.2f}")

    out.append("\nCaida de facil a dificil, AP_3D R40")
    for c in ("Car", "Pedestrian", "Cyclist"):
        v = [r2(x) for x in res[(c, "AP_R40", "3d")]]
        out.append(f"{c:<11} {v[0]:.2f} - {v[2]:.2f} = {r2(v[0] - v[2]):.2f}")

    (salida / "comparacion_ap.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print("\n" + "\n".join(lineas))


if __name__ == "__main__":
    main()
