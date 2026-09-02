"""Sondea los metadatos de varios datasets del hub sin descargar los datos.

``LeRobotDatasetMetadata`` baja solo la carpeta ``meta/`` (unos pocos KB), no
los vídeos ni los parquet. Sirve para decidir qué datasets merece la pena
descargar enteros antes de comprometer gigas de disco.
"""

from __future__ import annotations

import sys

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

CANDIDATES = [
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/svla_so101_pickplace",
    "lerobot/xarm_lift_medium",
]


def human_mb(n_bytes: float) -> str:
    return f"{n_bytes / 1e6:,.1f} MB"


def probe(repo_id: str) -> None:
    print(f"\n{'=' * 70}\n{repo_id}\n{'=' * 70}")
    try:
        meta = LeRobotDatasetMetadata(repo_id)
    except Exception as exc:  # noqa: BLE001 - queremos ver cualquier fallo
        print(f"  FALLA  {type(exc).__name__}: {str(exc)[:300]}")
        return

    print(f"  robot_type      {meta.robot_type}")
    print(f"  fps             {meta.fps}")
    print(f"  episodios       {meta.total_episodes}")
    print(f"  frames          {meta.total_frames:,}")
    print(f"  duracion aprox  {meta.total_frames / meta.fps / 60:.1f} min")
    print(f"  tareas          {list(meta.tasks.index)[:3]}")
    print(f"  camaras         {meta.camera_keys}")
    print("  features:")
    for key, spec in meta.features.items():
        print(f"    {key:34s} {spec['dtype']:8s} {tuple(spec['shape'])}")


def main() -> int:
    for repo_id in CANDIDATES:
        probe(repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
