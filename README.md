# Arranque en robótica e imitation learning, sin comprar hardware

Bitácora de aprendizaje: montar el pipeline completo de *imitation learning* con
[LeRobot](https://github.com/huggingface/lerobot), primero con datasets y
simuladores públicos, y después con el hardware que ya tenía en un cajón — un
ESP32, dos servos y una webcam.

Vengo de software, no de robótica. Cada concepto que uso está explicado en
[`docs/01-conceptos.md`](docs/01-conceptos.md) en castellano y sin dar por
supuesto nada del campo.

**Hallazgos honestos**: documento lo que falla igual que lo que funciona. Si algo
no llegó a funcionar, lo dice explícitamente en vez de dejar el hueco.

## Estructura

```
docs/        Bitácora y explicaciones conceptuales
firmware/    Sketch de Arduino para el ESP32 (pan/tilt de 2 servos)
python/      Teleoperación, grabación de datasets, utilidades
scripts/     Entrenamiento y evaluación
notebooks/   Exploración de datasets
```

## Estado

### Parte A — LeRobot sobre datasets y simulación públicos

| # | Tarea | Estado |
|---|---|---|
| 1 | Instalar LeRobot en Windows y documentar qué rompe | **hecho** — 8 problemas reales documentados |
| 2 | Explorar 2-3 datasets públicos de manipulación | **hecho** — pusht, SO-101 real, ALOHA |
| 3 | Entrenar ACT y evaluar en simulación | pipeline verificado de punta a punta; entrenamiento largo en curso |

### Parte B — Mi propio setup (ESP32 + 2 servos + webcam)

| # | Tarea | Estado |
|---|---|---|
| 4 | Plataforma pan/tilt con ESP32 hablando con Python | firmware y driver escritos — **sin probar en hardware** |
| 5 | Teleoperación por teclado grabando en formato LeRobot | escrito, y el **formato verificado** con una prueba de ida y vuelta sin hardware |
| 6 | 50-100 demostraciones de centrar un objeto de color | **bloqueado**: requiere la plataforma montada. Protocolo de grabación documentado |

Lo que está sin verificar está marcado como tal. No hay resultados inventados.

## Dónde vive cada cosa

Este repo contiene **solo código y documentación**. Todo lo pesado (entorno
virtual, cachés, datasets, checkpoints) vive fuera, en `D:\robotics-lab`, porque
el disco de sistema de esta máquina tiene 3 GB libres. El porqué y el cómo están
en [`docs/00-instalacion-windows.md`](docs/00-instalacion-windows.md).

## Documentos

- [`docs/00-instalacion-windows.md`](docs/00-instalacion-windows.md) — instalar
  LeRobot en Windows 11. ACLs de disco, cachés que llenan C:, torchcodec sin las
  DLLs de FFmpeg, el privilegio de symlinks. Con los errores literales.
- [`docs/01-conceptos.md`](docs/01-conceptos.md) — glosario de robótica e
  imitation learning para gente que viene de software.
- [`docs/02-datasets-lerobot.md`](docs/02-datasets-lerobot.md) — qué hay dentro
  de un `LeRobotDataset` v3.0, la relación entre `state` y `action`, y cómo el
  action chunking se convierte en una línea de código.
- [`docs/03-entrenamiento.md`](docs/03-entrenamiento.md) — entrenar ACT y
  evaluarla en simulación. Los cuatro tropiezos hasta que arrancó y dónde está
  de verdad el cuello de botella.
- [`docs/04-parte-b-pantilt.md`](docs/04-parte-b-pantilt.md) — el robot propio:
  montaje, firmware, protocolo serial y cómo grabar demostraciones que sirvan.

## Los cuatro errores silenciosos

Lo que más me ha costado del proyecto no son los errores que revientan, son los
que no. Están explicados en su sitio, pero los reúno aquí porque son el tipo de
fallo que produce un dataset que entrena perfectamente y no funciona:

1. **BGR vs RGB** — OpenCV da BGR, LeRobot espera RGB. Entrena igual, y el
   dataset queda inservible para cualquier otra cosa.
2. **El buffer de la webcam** — sin `CAP_PROP_BUFFERSIZE=1` emparejas una imagen
   de hace medio segundo con la posición actual del servo.
3. **El HUD quemado en la imagen** — la política aprende a leer el contador de
   frames en vez de a mirar el objeto, y valida estupendamente.
4. **Deriva del reloj** — `sleep(1/30)` acumula error y grabas a 27 Hz creyendo
   que grabas a 30. Los timestamps del dataset mienten.

Y uno mío, de análisis: normalizar `|action - state|` por la media absoluta da
conclusiones falsas cuando los datos están centrados en cero. Hay que normalizar
por el rango.
