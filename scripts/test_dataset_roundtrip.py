"""Verifica que el formato que escribe `record_pantilt.py` es un LeRobotDataset válido.

No hace falta hardware: simula el pan/tilt y la webcam, graba tres episodios
cortos con exactamente el mismo código de features y el mismo bucle que el
grabador real, y después **vuelve a abrir el dataset con LeRobotDataset** y
comprueba que todo cuadra.

Esto es lo que separa "escribí un dict con las claves correctas" de "el dataset
se puede entrenar". El escritor puede aceptar datos que el lector luego rechaza,
y la única forma de saberlo es cerrar el círculo.

Uso:
    python test_dataset_roundtrip.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from record_pantilt import FPS, build_features  # noqa: E402

WIDTH, HEIGHT = 160, 120
N_EPISODES = 3
N_FRAMES = 20
TASK = "centrar el objeto de color en el frame"

# Mismos límites que el firmware.
PAN_MIN, PAN_MAX = 20.0, 160.0
TILT_MIN, TILT_MAX = 45.0, 135.0


class FakePanTilt:
    """Imita el firmware: interpola hacia el objetivo con un límite de velocidad.

    Reproduce el slew rate limiter del `.ino` para que `state` vaya por detrás
    de `action` igual que en el hardware real. Si devolviera `state == action`,
    la prueba pasaría igualmente pero no estaría comprobando lo que importa.
    """

    def __init__(self, slew_dps: float = 180.0) -> None:
        self.pan = self.tilt = 90.0
        self.target_pan = self.target_tilt = 90.0
        self.max_step = slew_dps / FPS

    def _slew(self, cur: float, tgt: float) -> float:
        d = np.clip(tgt - cur, -self.max_step, self.max_step)
        return float(cur + d)

    def tick(self) -> None:
        self.pan = self._slew(self.pan, self.target_pan)
        self.tilt = self._slew(self.tilt, self.target_tilt)

    def move(self, pan: float, tilt: float) -> None:
        self.target_pan = float(np.clip(pan, PAN_MIN, PAN_MAX))
        self.target_tilt = float(np.clip(tilt, TILT_MIN, TILT_MAX))

    @property
    def state_vector(self) -> list[float]:
        return [self.pan, self.tilt]


def fake_frame(step: int, pan: float, tilt: float) -> np.ndarray:
    """Genera una imagen RGB sintética con un círculo cuya posición depende de la pose.

    No es un render realista y no pretende serlo: solo necesito píxeles que
    cambien de forma correlacionada con el estado, para que el vídeo resultante
    no sea un color plano que el codificador comprima a nada y enmascare un
    problema de tamaño.
    """
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :, 2] = 40  # fondo azulado

    cx = int(np.interp(pan, [PAN_MIN, PAN_MAX], [10, WIDTH - 10]))
    cy = int(np.interp(tilt, [TILT_MIN, TILT_MAX], [HEIGHT - 10, 10]))

    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 8**2
    img[mask] = (255, 60, 60)
    img[0, step % WIDTH] = (255, 255, 255)  # marca que cambia cada frame
    return img


def record(root: Path, repo_id: str) -> None:
    print(f"grabando {N_EPISODES} episodios de {N_FRAMES} frames en {root}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(HEIGHT, WIDTH),
        root=root,
        robot_type="esp32_pantilt",
        use_videos=True,
    )

    rng = np.random.default_rng(0)
    for ep in range(N_EPISODES):
        robot = FakePanTilt()
        target_pan, target_tilt = 90.0, 90.0

        for step in range(N_FRAMES):
            # 1. observación
            frame = fake_frame(step, robot.pan, robot.tilt)
            state = robot.state_vector

            # 2. "decisión humana" simulada: un paseo aleatorio suave
            target_pan = float(np.clip(target_pan + rng.normal(0, 3), PAN_MIN, PAN_MAX))
            target_tilt = float(np.clip(target_tilt + rng.normal(0, 3), TILT_MIN, TILT_MAX))

            # 3. actuar
            robot.move(target_pan, target_tilt)

            # 4. grabar: state de ANTES de moverse, action = objetivo nuevo
            dataset.add_frame({
                "observation.images.webcam": frame,
                "observation.state": np.array(state, dtype=np.float32),
                "action": np.array([target_pan, target_tilt], dtype=np.float32),
                "task": TASK,
            })

            robot.tick()

        dataset.save_episode()
        print(f"  episodio {ep} guardado")

    dataset.finalize()
    print("dataset cerrado")


def verify(root: Path, repo_id: str) -> bool:
    print("\nreabriendo el dataset con LeRobotDataset...")
    ds = LeRobotDataset(repo_id, root=root)
    ok = True

    def check(label: str, got: object, want: object) -> None:
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  [{'OK ' if good else 'MAL'}] {label:38s} {got}  (esperado {want})")

    check("episodios", ds.meta.total_episodes, N_EPISODES)
    check("frames", ds.meta.total_frames, N_EPISODES * N_FRAMES)
    check("fps", ds.meta.fps, FPS)
    check("robot_type", ds.meta.robot_type, "esp32_pantilt")
    check("len(dataset)", len(ds), N_EPISODES * N_FRAMES)

    sample = ds[0]
    check("shape de la imagen", tuple(sample["observation.images.webcam"].shape),
          (3, HEIGHT, WIDTH))
    check("shape de observation.state", tuple(sample["observation.state"].shape), (2,))
    check("shape de action", tuple(sample["action"].shape), (2,))
    check("task", sample["task"], TASK)

    img = sample["observation.images.webcam"]
    in_range = bool(img.min() >= 0.0 and img.max() <= 1.0)
    print(f"  [{'OK ' if in_range else 'MAL'}] imagen normalizada a [0,1]         "
          f"[{img.min():.3f}, {img.max():.3f}]")
    ok = ok and in_range

    # Lo que de verdad quería comprobar: que state persigue a action y no son
    # el mismo número. Si esto falla, el dataset no tiene dinámica que aprender.
    states = np.stack([ds[i]["observation.state"].numpy() for i in range(N_FRAMES)])
    actions = np.stack([ds[i]["action"].numpy() for i in range(N_FRAMES)])
    lag = float(np.abs(actions - states).mean())
    has_lag = lag > 1e-3
    print(f"  [{'OK ' if has_lag else 'MAL'}] state va por detras de action      "
          f"|action-state| medio = {lag:.4f}")
    ok = ok and has_lag

    # Ventana temporal: lo que necesita ACT.
    horizon = 8
    windowed = LeRobotDataset(repo_id, root=root,
                              delta_timestamps={"action": [i / FPS for i in range(horizon)]})
    shape = tuple(windowed[0]["action"].shape)
    good = shape == (horizon, 2)
    ok = ok and good
    print(f"  [{'OK ' if good else 'MAL'}] chunk de acciones para ACT           {shape}"
          f"  (esperado {(horizon, 2)})")

    print("\nestructura en disco generada:")
    for p in sorted(root.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(root).as_posix():58s} {p.stat().st_size / 1024:8.1f} KB")

    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pantilt_roundtrip_"))
    root = tmp / "dataset"
    repo_id = "test/pantilt-roundtrip"
    try:
        record(root, repo_id)
        ok = verify(root, repo_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODO CORRECTO: el formato es un LeRobotDataset válido."
                  if ok else "HAY FALLOS: revisa las líneas marcadas MAL."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
