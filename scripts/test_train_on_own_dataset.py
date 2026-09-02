"""Comprueba que `lerobot-train` entrena sobre un dataset generado por mi grabador.

`test_dataset_roundtrip.py` demuestra que el dataset se puede *leer*. Esto es un
paso más: que el pipeline de entrenamiento de LeRobot lo acepta de verdad. Son
cosas distintas — el lector es tolerante y el entrenador no: exige que las
features estén clasificadas correctamente como estado / visual / acción, que las
estadísticas de normalización existan y que el chunking encaje con la longitud
de los episodios.

Genera un dataset sintético con la misma forma que produciría el pan/tilt real,
lanza `lerobot-train` unos pocos pasos en CPU, y comprueba que sale con código 0
y deja un checkpoint.

No prueba que la política aprenda nada — con datos aleatorios no hay nada que
aprender. Prueba que el formato es entrenable.

Uso:
    python test_train_on_own_dataset.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from record_pantilt import FPS, build_features  # noqa: E402

WIDTH, HEIGHT = 96, 96
N_EPISODES = 8
# Los episodios tienen que ser mas largos que el chunk de ACT, o el dataset
# seria todo relleno. Con chunk_size=20 y 40 frames por episodio va sobrado.
N_FRAMES = 40
CHUNK = 20
TRAIN_STEPS = 4
TASK = "centrar el objeto de color en el frame"

PAN_MIN, PAN_MAX = 20.0, 160.0
TILT_MIN, TILT_MAX = 45.0, 135.0


def synth_frame(pan: float, tilt: float) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :, 2] = 40
    cx = int(np.interp(pan, [PAN_MIN, PAN_MAX], [8, WIDTH - 8]))
    cy = int(np.interp(tilt, [TILT_MIN, TILT_MAX], [HEIGHT - 8, 8]))
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 6**2] = (255, 60, 60)
    return img


def build_dataset(root: Path, repo_id: str) -> None:
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=build_features(HEIGHT, WIDTH),
        root=root, robot_type="esp32_pantilt", use_videos=True,
    )
    rng = np.random.default_rng(0)
    for _ in range(N_EPISODES):
        pan = tilt = 90.0
        tgt_pan = tgt_tilt = 90.0
        for _ in range(N_FRAMES):
            frame = synth_frame(pan, tilt)
            state = np.array([pan, tilt], dtype=np.float32)
            tgt_pan = float(np.clip(tgt_pan + rng.normal(0, 4), PAN_MIN, PAN_MAX))
            tgt_tilt = float(np.clip(tgt_tilt + rng.normal(0, 4), TILT_MIN, TILT_MAX))
            ds.add_frame({
                "observation.images.webcam": frame,
                "observation.state": state,
                "action": np.array([tgt_pan, tgt_tilt], dtype=np.float32),
                "task": TASK,
            })
            pan += np.clip(tgt_pan - pan, -6, 6)
            tilt += np.clip(tgt_tilt - tilt, -6, 6)
        ds.save_episode()
    ds.finalize()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pantilt_train_"))
    root = tmp / "dataset"
    out = tmp / "out"
    repo_id = "test/pantilt-trainable"

    try:
        print(f"generando {N_EPISODES}x{N_FRAMES} frames sinteticos...")
        build_dataset(root, repo_id)
        print(f"dataset listo en {root}\n")

        cmd = [
            sys.executable, "-m", "lerobot.scripts.lerobot_train",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={root}",
            "--policy.type=act",
            "--policy.push_to_hub=false",
            # CPU a proposito: esto es una prueba de formato, no de rendimiento,
            # y asi no compite por la GPU con un entrenamiento de verdad.
            "--policy.device=cpu",
            f"--policy.chunk_size={CHUNK}",
            f"--policy.n_action_steps={CHUNK}",
            f"--output_dir={out}",
            "--job_name=test_trainable",
            f"--steps={TRAIN_STEPS}",
            "--batch_size=2",
            # 0 workers: en Windows cada worker hace spawn de un interprete
            # entero, y para 4 pasos eso tarda mas que el entrenamiento.
            "--num_workers=0",
            f"--save_freq={TRAIN_STEPS}",
            "--log_freq=1",
            "--wandb.enable=false",
        ]
        print("lanzando lerobot-train...\n  " + " ".join(cmd[2:]) + "\n")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")

        ok = proc.returncode == 0
        ckpt = out / "checkpoints" / f"{TRAIN_STEPS:06d}" / "pretrained_model"
        has_ckpt = ckpt.is_dir()

        if not ok:
            print("lerobot-train fallo. Ultimas lineas:\n")
            print("\n".join((proc.stdout + proc.stderr).splitlines()[-25:]))
        else:
            for line in proc.stdout.splitlines():
                if "step:" in line and "loss:" in line:
                    print("  " + line.split("ot_train.py:")[-1].strip()[:120])

        print()
        print(f"  [{'OK ' if ok else 'MAL'}] lerobot-train sale con codigo 0")
        print(f"  [{'OK ' if has_ckpt else 'MAL'}] checkpoint escrito en {ckpt.name}")

        if ok and has_ckpt:
            files = sorted(p.name for p in ckpt.iterdir())
            print(f"        contenido: {', '.join(files)}")

        good = ok and has_ckpt
        print("\n" + ("CORRECTO: un dataset generado por record_pantilt.py es entrenable "
                      "con lerobot-train sin conversion." if good else
                      "FALLO: el formato no es entrenable tal cual."))
        return 0 if good else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
