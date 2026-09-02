"""Disecciona un LeRobotDataset: qué contiene, cómo está guardado, qué devuelve.

Uso:
    python explore_dataset.py lerobot/pusht
    python explore_dataset.py lerobot/pusht --skip-download   # solo metadatos

Responde a las preguntas del punto 2 de la Parte A:
  - ¿cuántas demostraciones hay?
  - ¿qué contiene cada frame?
  - ¿cómo está estructurado el formato LeRobotDataset en disco?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def sizeof(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n {title}\n{'=' * 78}")


def show_metadata(ds: LeRobotDataset) -> None:
    section("1. METADATOS — el resumen del dataset")
    m = ds.meta
    print(f"repo_id              {m.repo_id}")
    print(f"robot_type           {m.robot_type}")
    print(f"fps                  {m.fps} Hz")
    print(f"episodios (demos)    {m.total_episodes}")
    print(f"frames totales       {m.total_frames:,}")
    print(f"frames por episodio  {m.total_frames / m.total_episodes:.0f} de media")
    print(f"duración total       {m.total_frames / m.fps / 60:.1f} min")
    print(f"raíz en disco        {ds.root}")
    print("\ntareas (instrucción en lenguaje natural de cada episodio):")
    for task in list(m.tasks.index)[:5]:
        print(f"  - {task}")


def show_features(ds: LeRobotDataset) -> None:
    section("2. FEATURES — el esquema de un frame")
    print("Las 4 primeras son los datos de verdad; el resto es contabilidad que")
    print("LeRobot añade a todos los datasets para poder indexar y ordenar.\n")
    print(f"{'clave':36s} {'dtype':9s} {'shape':16s} nombres")
    print("-" * 78)
    for key, spec in ds.meta.features.items():
        names = spec.get("names")
        names_str = ", ".join(names) if isinstance(names, list) and len(names) <= 6 else ""
        print(f"{key:36s} {spec['dtype']:9s} {str(tuple(spec['shape'])):16s} {names_str}")


def show_one_frame(ds: LeRobotDataset) -> None:
    section("3. UN FRAME DE VERDAD — lo que devuelve dataset[0]")
    sample = ds[0]
    print("Un LeRobotDataset es un torch Dataset normal: indexas y te da un dict\n")
    print(f"{'clave':36s} {'tipo':16s} {'shape':20s} rango")
    print("-" * 100)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            rng = ""
            if value.numel() and value.dtype.is_floating_point:
                rng = f"[{value.min():.3f}, {value.max():.3f}]"
            elif value.numel():
                rng = f"[{value.min()}, {value.max()}]"
            print(f"{key:36s} {str(value.dtype):16s} {str(tuple(value.shape)):20s} {rng}")
        else:
            print(f"{key:36s} {type(value).__name__:16s} {'-':20s} {str(value)[:40]}")

    print("\nOJO con las imágenes: llegan como float32 en [0,1] y en formato")
    print("CHW (canal, alto, ancho), no HWC como las da OpenCV. LeRobot ya hace")
    print("esa conversión al leer el vídeo — no la repitas tú.")


def show_state_action(ds: LeRobotDataset) -> None:
    section("4. STATE vs ACTION — la relación que hay que entender")
    n = min(2000, len(ds))
    states, actions = [], []
    for i in range(0, n, max(1, n // 200)):
        f = ds[i]
        states.append(f["observation.state"].numpy())
        actions.append(f["action"].numpy())
    s = np.stack(states)
    a = np.stack(actions)

    print(f"muestreados {len(s)} frames\n")
    print(f"{'dim':5s} {'state min':>11s} {'state max':>11s} {'action min':>11s} {'action max':>11s}")
    print("-" * 55)
    names = ds.meta.features["observation.state"].get("names") or []
    for d in range(s.shape[1]):
        label = names[d] if d < len(names) else str(d)
        print(f"{str(label)[:5]:5s} {s[:, d].min():11.3f} {s[:, d].max():11.3f} "
              f"{a[:, d].min():11.3f} {a[:, d].max():11.3f}")

    if s.shape == a.shape:
        diff = np.abs(a - s).mean()
        scale = np.abs(s).mean() + 1e-8
        print(f"\n|action - state| medio: {diff:.4f}  ({100 * diff / scale:.1f}% de la escala del estado)")
        if diff / scale < 0.15:
            print("-> action y state viven en el MISMO espacio: la acción es una")
            print("   posición objetivo de las articulaciones, no una velocidad ni")
            print("   un par. Es lo normal en LeRobot (ver docs/01-conceptos.md).")
        else:
            print("-> difieren bastante: o la acción está en otro espacio (velocidad,")
            print("   cartesiano) o el seguimiento del controlador es malo.")


def show_episodes(ds: LeRobotDataset) -> None:
    section("5. EPISODIOS — dónde empieza y acaba cada demostración")
    print("Los frames de todos los episodios están concatenados en un solo índice")
    print("plano. `episode_data_index` guarda los cortes.\n")
    starts = ds.episode_data_index["from"][:8].tolist()
    ends = ds.episode_data_index["to"][:8].tolist()
    print(f"{'episodio':>9s} {'desde':>9s} {'hasta':>9s} {'frames':>8s} {'segundos':>9s}")
    print("-" * 50)
    for i, (a, b) in enumerate(zip(starts, ends)):
        print(f"{i:9d} {a:9d} {b:9d} {b - a:8d} {(b - a) / ds.meta.fps:9.1f}")
    lengths = (ds.episode_data_index["to"] - ds.episode_data_index["from"]).float()
    print(f"\nlongitud de episodio: min {lengths.min():.0f}, media {lengths.mean():.0f}, "
          f"max {lengths.max():.0f} frames")
    print("La variación importa: si unas demos duran el triple que otras, o la")
    print("tarea es de dificultad variable, o la teleoperación fue inconsistente.")


def show_stats(ds: LeRobotDataset) -> None:
    section("6. ESTADÍSTICAS — para qué sirve meta/stats.json")
    print("LeRobot precalcula media/desviación/min/max de cada feature sobre todo")
    print("el dataset. La política las usa para normalizar las entradas: sin eso,")
    print("una articulación que va de 0 a 3000 dominaría el gradiente frente a")
    print("otra que va de -1 a 1.\n")
    for key in ("observation.state", "action"):
        st = ds.meta.stats.get(key)
        if st is None:
            continue
        print(f"{key}")
        print(f"  mean  {np.array2string(np.asarray(st['mean']), precision=3, max_line_width=70)}")
        print(f"  std   {np.array2string(np.asarray(st['std']), precision=3, max_line_width=70)}")


def show_layout(ds: LeRobotDataset) -> None:
    section("7. ESTRUCTURA EN DISCO — el formato LeRobotDataset v3.0")
    root = Path(ds.root)
    if not root.exists():
        print(f"(no descargado en {root})")
        return

    print("Formato v3.0. Ojo: la mayoría de tutoriales describen el v2.1, que")
    print("guardaba un parquet y un mp4 POR EPISODIO. El v3.0 agrupa muchos")
    print("episodios en ficheros grandes (~100 MB datos, ~200 MB vídeo) porque")
    print("miles de ficheros pequeños hacen lentísimo tanto el hub como el disco.\n")

    for sub in ("meta", "data", "videos"):
        d = root / sub
        if not d.is_dir():
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        print(f"{sub}/  — {len(files)} ficheros, {sizeof(total)}")
        for p in files[:6]:
            print(f"    {p.relative_to(root).as_posix():58s} {sizeof(p.stat().st_size):>10s}")
        if len(files) > 6:
            print(f"    ... y {len(files) - 6} más")
        print()

    print("Qué hace cada parte:")
    print("  meta/info.json       fps, features, rutas, totales")
    print("  meta/stats.json      media/std/min/max para normalizar")
    print("  meta/tasks.parquet   las instrucciones en lenguaje natural")
    print("  meta/episodes/       índice de episodios: dónde vive cada uno")
    print("  data/chunk-XXX/      los datos NO visuales, en parquet")
    print("  videos/<cam>/        las imágenes, codificadas en mp4")
    print("\nLa decisión de diseño clave: las imágenes NO van en el parquet, van")
    print("en mp4. Un dataset de 25.000 frames a 480x640 son ~23 GB en PNG y unos")
    print("cientos de MB en vídeo. El coste es que hay que decodificar al vuelo —")
    print("por eso torchcodec es una dependencia crítica y no un adorno.")


def show_temporal_window(repo_id: str, ds: LeRobotDataset, root: Path | None) -> None:
    section("8. VENTANAS TEMPORALES — cómo se alimenta a ACT y Diffusion Policy")
    print("Estas políticas no consumen un frame suelto: necesitan una secuencia de")
    print("acciones futuras. `delta_timestamps` le pide al dataset que, para cada")
    print("índice, devuelva también frames desplazados en el tiempo.\n")

    fps = ds.meta.fps
    horizon = 16
    delta = {"action": [i / fps for i in range(horizon)]}
    windowed = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
    sample = windowed[0]

    print(f"sin delta_timestamps:  action shape = {tuple(ds[0]['action'].shape)}")
    print(f"con horizonte de {horizon}:   action shape = {tuple(sample['action'].shape)}")
    print("\nO sea que ahora cada elemento del dataset trae las 16 acciones")
    print("siguientes. Eso es exactamente el 'action chunking' de ACT: el modelo")
    print("aprende a predecir un bloque entero de golpe en vez de un paso.")
    if "action_is_pad" in sample:
        print(f"\naction_is_pad: {sample['action_is_pad'].tolist()}")
        print("Al final de un episodio no hay 16 acciones futuras, así que LeRobot")
        print("rellena y marca cuáles son relleno para que la pérdida las ignore.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--root", default=None)
    parser.add_argument("--skip-download", action="store_true",
                        help="solo metadatos, no baja vídeos ni parquet")
    args = parser.parse_args()

    root = Path(args.root) if args.root else None

    print(f"\n### {args.repo_id}")
    ds = LeRobotDataset(args.repo_id, root=root)

    show_metadata(ds)
    show_features(ds)

    if args.skip_download:
        print("\n(--skip-download: me salto todo lo que necesita los datos)")
        return 0

    show_one_frame(ds)
    show_state_action(ds)
    show_episodes(ds)
    show_stats(ds)
    show_layout(ds)
    show_temporal_window(args.repo_id, ds, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
