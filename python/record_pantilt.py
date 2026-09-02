"""Teleoperación por teclado del pan/tilt, grabando en formato LeRobotDataset.

Punto 5 de la Parte B. Controlas los dos servos con el teclado mientras la
webcam graba, y cada paso se guarda como un frame con:

    observation.images.webcam  imagen de la webcam
    observation.state          [pan, tilt] reales, leídos del firmware
    action                     [pan, tilt] objetivo, o sea lo que pulsaste

Al escribir directamente en formato LeRobotDataset v3.0, el dataset resultante
se puede entrenar con ``lerobot-train`` sin ninguna conversión, exactamente
igual que uno del hub.

Uso:
    python record_pantilt.py --repo-id maria/pantilt-centrar --port COM5
    python record_pantilt.py --repo-id maria/pantilt-centrar   # autodetecta

Teclado (con la ventana de vídeo enfocada):
    flechas / WASD   mover pan y tilt
    ESPACIO          guardar el episodio y preparar el siguiente
    R                descartar el episodio actual y repetirlo
    H                volver al centro
    Q / ESC          terminar la sesión
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_pantilt.pantilt import PanTilt, PanTiltError, find_esp32_port

logger = logging.getLogger(__name__)

# Frecuencia de grabación. 30 Hz es el estándar de facto en LeRobot y coincide
# con lo que da una webcam normal. Ver docs/03 sobre el presupuesto de tiempo
# por frame.
FPS = 30

# Cuánto se mueve el objetivo por cada pulsación de tecla, en grados. Es el
# hiperparámetro más importante de la teleoperación: demasiado grande y los
# movimientos salen a saltos (malas demostraciones); demasiado pequeño y no
# llegas a tiempo a seguir el objeto.
STEP_DEG = 1.5

# Límites, duplicados a propósito respecto al firmware. El firmware es la
# defensa real del hardware; esto es solo para que la vista previa no muestre un
# objetivo imposible.
PAN_MIN, PAN_MAX = 20.0, 160.0
TILT_MIN, TILT_MAX = 45.0, 135.0

WINDOW = "pan/tilt teleop  —  flechas/WASD mover · ESPACIO guardar · R repetir · Q salir"

# Códigos de tecla de cv2.waitKey. Las flechas dan códigos distintos según la
# plataforma, así que ofrecemos WASD como alternativa fiable en todas.
KEY_ESC = 27
KEY_SPACE = 32


def build_features(height: int, width: int) -> dict:
    """Especificación de features del dataset.

    El formato lo dicta ``lerobot.utils.feature_utils.hw_to_dataset_features``;
    lo escribo explícito aquí para que se vea qué contiene un frame en vez de
    esconderlo tras un helper.

    Args:
        height: Alto del frame de la webcam, en píxeles.
        width: Ancho del frame de la webcam, en píxeles.

    Returns:
        Dict de features listo para ``LeRobotDataset.create``.
    """
    return {
        "observation.images.webcam": {
            # "video" = se codifica a mp4 al cerrar el episodio. La alternativa
            # es "image" (PNG sueltos), que ocupa un orden de magnitud más.
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["pan", "tilt"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["pan", "tilt"],
        },
    }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_key(key: int, target_pan: float, target_tilt: float) -> tuple[float, float]:
    """Traduce una pulsación a un nuevo objetivo.

    Devuelve el objetivo sin cambios si la tecla no es de movimiento, de modo
    que "no pulsar nada" es una acción válida y se graba igual. Esto importa:
    los frames en los que el humano decide *no* moverse son parte de la
    demostración, no huecos.
    """
    dp = dt = 0.0

    if key in (ord("a"), ord("A"), 81, 2424832):  # izquierda
        dp = -STEP_DEG
    elif key in (ord("d"), ord("D"), 83, 2555904):  # derecha
        dp = +STEP_DEG
    elif key in (ord("w"), ord("W"), 82, 2490368):  # arriba
        dt = +STEP_DEG
    elif key in (ord("s"), ord("S"), 84, 2621440):  # abajo
        dt = -STEP_DEG

    return (
        clamp(target_pan + dp, PAN_MIN, PAN_MAX),
        clamp(target_tilt + dt, TILT_MIN, TILT_MAX),
    )


def draw_hud(
    frame: np.ndarray,
    episode: int,
    n_frames: int,
    state: tuple[float, float],
    action: tuple[float, float],
    dropped: int,
) -> np.ndarray:
    """Superpone información de la sesión. Solo para la vista previa.

    Importante: el HUD se dibuja sobre una **copia**. Si lo pintáramos sobre el
    frame que guardamos, el dataset tendría texto quemado en las imágenes y la
    política aprendería a leer el contador de frames en vez de a mirar el
    objeto. Es un error fácil de cometer y difícil de detectar después.
    """
    canvas = frame.copy()
    lines = [
        f"episodio {episode}   frames {n_frames}",
        f"state  pan={state[0]:6.1f}  tilt={state[1]:6.1f}",
        f"action pan={action[0]:6.1f}  tilt={action[1]:6.1f}",
    ]
    if dropped:
        lines.append(f"frames tarde: {dropped}")

    for i, text in enumerate(lines):
        y = 22 + i * 20
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Abre la webcam y comprueba que de verdad da la resolución pedida.

    En Windows conviene forzar el backend DSHOW: el backend por defecto (MSMF)
    tarda varios segundos en abrir la cámara y en muchas webcams ignora
    ``CAP_PROP_FRAME_WIDTH``.

    Raises:
        RuntimeError: Si la cámara no abre o no da imagen.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"No pude abrir la webcam {index}. Prueba otro índice con --camera, "
            "y cierra cualquier otra app que la esté usando."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Buffer de 1: sin esto, cv2 acumula frames y acabas grabando imágenes de
    # hace medio segundo emparejadas con el estado actual del servo. El dataset
    # queda desincronizado y la política no puede aprender nada.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"La webcam {index} abrió pero no da imagen.")

    actual_h, actual_w = frame.shape[:2]
    if (actual_w, actual_h) != (width, height):
        logger.warning(
            "La webcam ignoró la resolución pedida: pedí %dx%d, da %dx%d. "
            "Uso la real; el dataset será consistente igualmente.",
            width, height, actual_w, actual_h,
        )
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="p.ej. maria/pantilt-centrar")
    parser.add_argument("--port", default=None, help="puerto COM del ESP32 (autodetecta si se omite)")
    parser.add_argument("--camera", type=int, default=0, help="índice de la webcam")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--root", default=None, help="carpeta del dataset (por defecto HF_LEROBOT_HOME)")
    parser.add_argument(
        "--task",
        default="centrar el objeto de color en el frame",
        help="descripción en lenguaje natural de la tarea, se guarda en cada frame",
    )
    parser.add_argument("--resume", action="store_true", help="añadir episodios a un dataset ya existente")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    port = args.port or find_esp32_port()
    if port is None:
        logger.error(
            "No pude autodetectar el ESP32. Pasa --port COM5 (míralo en el "
            "Administrador de dispositivos, en Puertos COM y LPT)."
        )
        return 1

    cap = open_camera(args.camera, args.width, args.height)
    ok, probe = cap.read()
    height, width = probe.shape[:2]

    root = Path(args.root) if args.root else None

    try:
        with PanTilt(port) as robot:
            logger.info("ESP32 conectado en %s, firmware %s", port, robot.firmware_version)
            state = robot.home()

            if args.resume:
                dataset = LeRobotDataset(args.repo_id, root=root)
                dataset.start_recording()
                logger.info("Continuando dataset con %d episodios", dataset.num_episodes)
            else:
                dataset = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    fps=FPS,
                    features=build_features(height, width),
                    root=root,
                    robot_type="esp32_pantilt",
                    use_videos=True,
                    # Escribir imágenes en hilos aparte: si no, la codificación
                    # bloquea el bucle y se pierden frames.
                    image_writer_threads=4,
                )

            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            episode = dataset.num_episodes
            n_frames = 0
            dropped = 0
            target_pan, target_tilt = state.target_pan, state.target_tilt
            period = 1.0 / FPS
            next_tick = time.perf_counter()

            logger.info("Grabando. ESPACIO guarda el episodio, R lo repite, Q sale.")

            while True:
                # --- 1. observación: imagen + estado, lo más juntos posible ---
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Frame de webcam perdido, lo salto")
                    continue
                state = robot.query()

                # --- 2. decisión humana: leer teclado -----------------------
                # waitKey(1) es también lo que refresca la ventana de cv2; sin
                # una llamada a waitKey la imagen nunca se pinta.
                key = cv2.waitKey(1) & 0xFFFFFF

                if key in (ord("q"), ord("Q"), KEY_ESC):
                    logger.info("Salida pedida. Descarto el episodio en curso (%d frames).", n_frames)
                    dataset.clear_episode_buffer()
                    break

                if key in (ord("h"), ord("H")):
                    state = robot.home()
                    target_pan, target_tilt = state.target_pan, state.target_tilt

                if key == KEY_SPACE:
                    if n_frames == 0:
                        logger.warning("Episodio vacío, no guardo nada.")
                    else:
                        dataset.save_episode()
                        logger.info("Episodio %d guardado (%d frames)", episode, n_frames)
                        episode += 1
                    n_frames, dropped = 0, 0
                    state = robot.home()
                    target_pan, target_tilt = state.target_pan, state.target_tilt
                    next_tick = time.perf_counter()
                    continue

                if key in (ord("r"), ord("R")):
                    dataset.clear_episode_buffer()
                    logger.info("Episodio descartado (%d frames)", n_frames)
                    n_frames, dropped = 0, 0
                    state = robot.home()
                    target_pan, target_tilt = state.target_pan, state.target_tilt
                    next_tick = time.perf_counter()
                    continue

                target_pan, target_tilt = apply_key(key, target_pan, target_tilt)

                # --- 3. actuar ----------------------------------------------
                robot.move(target_pan, target_tilt)

                # --- 4. grabar el frame -------------------------------------
                # OpenCV entrega BGR; LeRobot espera RGB. Si se te olvida esta
                # conversión el dataset entrena igual (la red aprende con los
                # canales cambiados) pero en cuanto veas los vídeos o mezcles
                # con otro dataset todo estará azul. Fallo silencioso clásico.
                dataset.add_frame({
                    "observation.images.webcam": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    "observation.state": np.array(state.state_vector, dtype=np.float32),
                    "action": np.array([target_pan, target_tilt], dtype=np.float32),
                    "task": args.task,
                })
                n_frames += 1

                cv2.imshow(
                    WINDOW,
                    draw_hud(frame, episode, n_frames, state.state_vector,
                             (target_pan, target_tilt), dropped),
                )

                # --- 5. mantener el ritmo -----------------------------------
                # Reloj absoluto en vez de sleep(period): con sleep el error de
                # cada iteración se acumula y acabas grabando a 27 Hz creyendo
                # que grabas a 30, con lo que los timestamps del dataset mienten.
                next_tick += period
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    # Vamos tarde: reengancho el reloj para no entrar en espiral.
                    dropped += 1
                    next_tick = time.perf_counter()

            dataset.finalize()
            logger.info("Dataset cerrado: %d episodios en %s", dataset.num_episodes, dataset.root)

    except PanTiltError as exc:
        logger.error("Fallo de comunicación con la placa: %s", exc)
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
