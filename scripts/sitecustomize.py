"""Registra el directorio de DLLs de FFmpeg para que torchcodec pueda cargarlas.

Copiar a ``<venv>\\Lib\\site-packages\\sitecustomize.py``. Python importa este
módulo automáticamente al arrancar el intérprete (lo hace el módulo ``site`` de
la stdlib), así que el arreglo aplica a todo proceso del entorno —
``lerobot-train``, notebooks, scripts propios — sin tener que acordarse de nada.

EL PROBLEMA
-----------
``torchcodec`` es el decodificador de vídeo que usa LeRobot para leer los mp4 de
los datasets. Trae sus propias DLLs (``libtorchcodec_core4..8.dll``) pero estas
dependen de las DLLs de FFmpeg (``avcodec``, ``avformat``, ``avutil``,
``swscale``, ``swresample``), que en Windows no vienen con el sistema.

Lo natural es descargar FFmpeg y añadirlo al ``PATH``. **No funciona.** Desde
Python 3.8, en Windows, ``PATH`` dejó de usarse para resolver las dependencias
de las DLLs de los módulos de extensión — fue un cambio de seguridad
deliberado (ver ``os.add_dll_directory`` en la documentación de la stdlib). El
error que da es especialmente malo:

    FileNotFoundError: Could not find module 'libtorchcodec_core4.dll'
    (or one of its dependencies)

Dice que no encuentra ``libtorchcodec_core4.dll``, que **sí** está ahí. La que
falta es una de sus dependencias, y el mensaje no dice cuál.

Nota aparte: torchcodec prueba las versiones de FFmpeg de la 8 a la 4 y solo
reporta si fallan todas, así que en el traceback ves cinco errores encadenados
y el relevante es únicamente el de la versión que tienes instalada.
"""

import os
import sys
from pathlib import Path

# Se puede sobrescribir con la variable de entorno, útil si mueves FFmpeg.
_DEFAULT_FFMPEG_BIN = Path(r"D:\robotics-lab\ffmpeg\bin")


def _register_ffmpeg_dlls() -> None:
    if sys.platform != "win32":
        return  # En Linux y macOS el enlazador sí usa las rutas del sistema.

    raw = os.environ.get("LEROBOT_FFMPEG_BIN")
    ffmpeg_bin = Path(raw) if raw else _DEFAULT_FFMPEG_BIN

    if not ffmpeg_bin.is_dir():
        # Silencio a propósito: si alguien copia este venv a una máquina sin
        # FFmpeg, que falle luego torchcodec con su mensaje, y no cada arranque
        # del intérprete con un warning nuestro.
        return

    try:
        os.add_dll_directory(str(ffmpeg_bin))
    except OSError:
        pass


_register_ffmpeg_dlls()
