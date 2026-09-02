"""Extrae la curva de entrenamiento del log de `lerobot-train`.

El log mezcla la barra de progreso de tqdm (que usa retornos de carro) con las
líneas INFO, así que un `grep` normal devuelve resultados engañosos: parece que
solo hay unas pocas entradas cuando en realidad están todas pegadas al final de
las líneas de tqdm.

Uso:
    python parse_train_log.py D:\\robotics-lab\\logs_act_pusht.txt
    python parse_train_log.py D:\\robotics-lab\\logs_act_pusht.txt --markdown
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# El campo `step` sale abreviado ("5K", "12K") cuando pasa de mil, así que no
# sirve para ordenar. Uso `smpl` (muestras vistas) dividido por el batch, o
# mejor: me quedo con el step tal cual para mostrar y ordeno por orden de
# aparición, que en un log secuencial es lo mismo.
LINE_RE = re.compile(
    r"step:(?P<step>[\dKM.]+)\s+"
    r"smpl:(?P<smpl>[\dKM.]+)\s+"
    r"ep:(?P<ep>[\dKM.]+)\s+"
    r"epch:(?P<epch>[\d.]+)\s+"
    r"loss:(?P<loss>[\d.]+)\s+"
    r"grdn:(?P<grdn>[\d.]+)"
)
# lerobot imprime las métricas como repr de dict de Python (comillas simples),
# pero acepto también comillas dobles por si cambia a JSON en alguna versión.
EVAL_RE = re.compile(r"['\"]pc_success['\"]:\s*(?P<pc>[\d.]+)")
L1_RE = re.compile(r"l1_loss:(?P<l1>[\d.]+)")
KLD_RE = re.compile(r"kld_loss:(?P<kld>[\d.]+)")


def parse(path: Path) -> tuple[list[dict], list[float]]:
    # tqdm usa '\r' para reescribir la línea. Sin convertirlo a '\n' las
    # entradas INFO quedan escondidas dentro de líneas larguísimas.
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")

    rows: list[dict] = []
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        row = m.groupdict()
        l1 = L1_RE.search(line)
        kld = KLD_RE.search(line)
        row["l1"] = l1.group("l1") if l1 else ""
        row["kld"] = kld.group("kld") if kld else ""
        rows.append(row)

    evals = [float(m.group("pc")) for m in EVAL_RE.finditer(text)]
    return rows, evals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--markdown", action="store_true", help="salida como tabla markdown")
    ap.add_argument("--every", type=int, default=1, help="muestra 1 de cada N filas")
    args = ap.parse_args()

    rows, evals = parse(args.log)
    if not rows:
        print("No encontré ninguna línea de progreso. ¿Es el log correcto?")
        return 1

    shown = rows[:: args.every]
    if shown[-1] is not rows[-1]:
        shown.append(rows[-1])  # el último punto siempre interesa

    if args.markdown:
        print("| paso | época | pérdida | l1 | kld | grad norm |")
        print("|---|---|---|---|---|---|")
        for r in shown:
            print(f"| {r['step']} | {r['epch']} | {r['loss']} | {r['l1']} | "
                  f"{r['kld']} | {r['grdn']} |")
    else:
        print(f"{'paso':>8s} {'época':>7s} {'pérdida':>9s} {'l1':>8s} {'kld':>8s} {'grad':>9s}")
        print("-" * 55)
        for r in shown:
            print(f"{r['step']:>8s} {r['epch']:>7s} {r['loss']:>9s} {r['l1']:>8s} "
                  f"{r['kld']:>8s} {r['grdn']:>9s}")

    print(f"\n{len(rows)} puntos de log")
    print(f"pérdida: {rows[0]['loss']} (inicio) -> {rows[-1]['loss']} (último)")
    if evals:
        print(f"evaluaciones en simulación (pc_success): "
              f"{', '.join(f'{v:.1%}' for v in evals)}")
    else:
        print("evaluaciones en simulación: ninguna todavía")
    return 0


if __name__ == "__main__":
    sys.exit(main())
