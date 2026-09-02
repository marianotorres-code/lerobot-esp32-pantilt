"""Cliente serial de la plataforma pan/tilt sobre ESP32.

Habla el protocolo ASCII por línea definido en
``firmware/esp32_pantilt/esp32_pantilt.ino``.

Concepto clave (ver docs/01-conceptos.md): este driver distingue entre

    target  = lo que le pedimos al robot   -> se guarda como ``action``
    current = dónde está el robot de verdad -> se guarda como ``observation.state``

Ambos vienen en la misma respuesta del firmware, así que un solo intercambio
serial da los dos números que necesita un frame del dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports

# Debe coincidir con Serial.begin() del firmware.
BAUDRATE = 115200

# Tras abrir el puerto, el ESP32 se reinicia (ver nota en connect()) y tarda en
# arrancar. Este es el margen que damos a que aparezca la línea READY.
BOOT_TIMEOUT_S = 5.0

# Timeout de una respuesta a un comando ya con la placa arrancada. El firmware
# contesta en microsegundos; si en 1 s no hay nada, algo va mal de verdad.
REPLY_TIMEOUT_S = 1.0


class PanTiltError(RuntimeError):
    """Fallo de comunicación o respuesta inesperada del firmware."""


@dataclass(frozen=True)
class PanTiltState:
    """Una lectura del firmware.

    Attributes:
        pan: Posición real del servo de pan, en grados.
        tilt: Posición real del servo de tilt, en grados.
        target_pan: Posición objetivo de pan, en grados.
        target_tilt: Posición objetivo de tilt, en grados.
        board_ms: ``millis()`` de la placa. Sirve para detectar reinicios
            inesperados (si baja de golpe, la placa se reseteó) y para medir
            jitter, pero NO para timestampear el dataset: ese reloj es el del
            host, que es el mismo que timestampea la webcam.
    """

    pan: float
    tilt: float
    target_pan: float
    target_tilt: float
    board_ms: int

    @property
    def state_vector(self) -> list[float]:
        """``observation.state``: dónde está el robot realmente."""
        return [self.pan, self.tilt]

    @property
    def action_vector(self) -> list[float]:
        """``action``: dónde le pedimos que estuviera."""
        return [self.target_pan, self.target_tilt]


def find_esp32_port() -> str | None:
    """Intenta adivinar el puerto COM del ESP32 por el chip USB-serial.

    Las placas ESP32 usan casi siempre un CP210x (Silicon Labs) o un CH340
    (WCH). Buscamos por VID de fabricante, que es más fiable que por
    descripción — la descripción cambia según el driver instalado.

    Returns:
        El nombre del puerto (``'COM5'``), o ``None`` si hay cero o varios
        candidatos. Con varios devolvemos ``None`` a propósito: elegir uno al
        azar y mover el servo equivocado es peor que pedir el puerto explícito.
    """
    known_vids = {
        0x10C4,  # Silicon Labs CP210x
        0x1A86,  # WCH CH340 / CH9102
        0x0403,  # FTDI
        0x303A,  # Espressif (USB nativo del ESP32-S2/S3/C3)
    }
    candidates = [p.device for p in list_ports.comports() if p.vid in known_vids]
    return candidates[0] if len(candidates) == 1 else None


class PanTilt:
    """Conexión con la plataforma pan/tilt.

    Uso::

        with PanTilt("COM5") as robot:
            robot.home()
            state = robot.move(100.0, 85.0)
            print(state.state_vector, state.action_vector)
    """

    def __init__(self, port: str, baudrate: int = BAUDRATE) -> None:
        self.port = port
        self.baudrate = baudrate
        self._ser: serial.Serial | None = None
        self.firmware_version: str | None = None

    # ------------------------------------------------------------ conexión --

    def connect(self) -> PanTiltState:
        """Abre el puerto y espera a que la placa arranque.

        Trampa de Windows/ESP32: al abrir el puerto serie, el driver activa las
        líneas DTR y RTS, que en las placas de desarrollo están cableadas al
        circuito de auto-reset (EN y GPIO0). O sea que **abrir el puerto
        reinicia el ESP32**. No es un bug: es el mecanismo que usa el IDE de
        Arduino para entrar en modo flash.

        La consecuencia práctica es que no se puede abrir el puerto y mandar un
        comando inmediatamente — se pierde, porque la placa está en el
        bootloader. Hay que esperar a la línea ``READY``, que es exactamente lo
        que hace este método.

        Returns:
            El estado inicial de la placa tras arrancar.

        Raises:
            PanTiltError: Si no llega ``READY`` dentro de ``BOOT_TIMEOUT_S``.
        """
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=REPLY_TIMEOUT_S,
            write_timeout=REPLY_TIMEOUT_S,
        )

        # Descarta el ruido del arranque (el ESP32 escupe el log del bootloader
        # a 74880 baudios, que a 115200 se lee como basura).
        self._ser.reset_input_buffer()

        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("READY"):
                parts = line.split()
                self.firmware_version = parts[3] if len(parts) > 3 else "desconocida"
                return self.query()

        raise PanTiltError(
            f"La placa en {self.port} no mandó READY en {BOOT_TIMEOUT_S:.0f} s. "
            "Comprueba que el firmware está flasheado y que ningún otro programa "
            "tiene el puerto abierto (el Monitor Serie del IDE de Arduino es el "
            "culpable habitual — solo un proceso puede tener el COM abierto)."
        )

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> "PanTilt":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------------- interno --

    def _require_open(self) -> serial.Serial:
        if self._ser is None or not self._ser.is_open:
            raise PanTiltError("El puerto no está abierto. Llama a connect() primero.")
        return self._ser

    def _readline(self) -> str:
        ser = self._require_open()
        # errors='replace' en vez de dejar que reviente: un byte corrupto por
        # ruido eléctrico no debe tirar abajo una sesión de grabación de 20
        # minutos. La línea corrupta fallará al parsear más arriba, que es un
        # fallo mucho más informativo que un UnicodeDecodeError.
        return ser.readline().decode("ascii", errors="replace").strip()

    def _command(self, cmd: str) -> PanTiltState:
        """Manda un comando y parsea la línea de estado que devuelve."""
        ser = self._require_open()
        ser.write((cmd + "\n").encode("ascii"))
        ser.flush()

        line = self._readline()
        if line.startswith("ERR"):
            raise PanTiltError(f"El firmware rechazó {cmd!r}: {line}")
        if not line.startswith("S "):
            raise PanTiltError(
                f"Respuesta inesperada a {cmd!r}: {line!r}. "
                "Suele significar que el host y el firmware están desincronizados "
                "(hay respuestas viejas en el buffer)."
            )

        parts = line.split()
        if len(parts) != 6:
            raise PanTiltError(f"Línea de estado malformada: {line!r}")

        try:
            return PanTiltState(
                pan=float(parts[1]),
                tilt=float(parts[2]),
                target_pan=float(parts[3]),
                target_tilt=float(parts[4]),
                board_ms=int(parts[5]),
            )
        except ValueError as exc:
            raise PanTiltError(f"No pude parsear la línea de estado {line!r}") from exc

    # ------------------------------------------------------------- comandos --

    def move(self, pan: float, tilt: float) -> PanTiltState:
        """Fija la posición objetivo y devuelve el estado inmediatamente después.

        Nota importante para el dataset: el ``state`` que devuelve es el de
        *antes* de que el servo se mueva, porque el firmware contesta al
        instante. Eso es lo correcto — el frame que grabamos es "estaba aquí, y
        le pedí ir allá".
        """
        return self._command(f"M {pan:.2f} {tilt:.2f}")

    def query(self) -> PanTiltState:
        """Lee el estado sin cambiar el objetivo."""
        return self._command("S")

    def home(self) -> PanTiltState:
        """Vuelve al centro (90, 90)."""
        return self._command("H")

    def set_speed(self, degrees_per_second: float) -> PanTiltState:
        """Cambia la velocidad máxima de slew del firmware."""
        return self._command(f"V {degrees_per_second:.2f}")

    def set_enabled(self, enabled: bool) -> PanTiltState:
        """Activa o suelta los servos.

        Soltarlos entre demostraciones evita que zumben y se calienten.
        """
        return self._command(f"E {1 if enabled else 0}")

    def ping(self) -> bool:
        """Comprueba que la placa responde."""
        ser = self._require_open()
        ser.write(b"P\n")
        ser.flush()
        return self._readline() == "PONG"
