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
| 1 | Instalar LeRobot en Windows y documentar qué rompe | en curso |
| 2 | Explorar 2-3 datasets públicos de manipulación | pendiente |
| 3 | Entrenar ACT / Diffusion Policy y evaluar en simulación | pendiente |

### Parte B — Mi propio setup (ESP32 + 2 servos + webcam)

| # | Tarea | Estado |
|---|---|---|
| 4 | Plataforma pan/tilt con ESP32 hablando con Python | firmware escrito, sin probar en hardware |
| 5 | Teleoperación por teclado grabando en formato LeRobot | pendiente |
| 6 | 50-100 demostraciones de centrar un objeto de color | pendiente |

## Dónde vive cada cosa

Este repo contiene **solo código y documentación**. Todo lo pesado (entorno
virtual, cachés, datasets, checkpoints) vive fuera, en `D:\robotics-lab`, porque
el disco de sistema de esta máquina tiene 3 GB libres. El porqué y el cómo están
en [`docs/00-instalacion-windows.md`](docs/00-instalacion-windows.md).

## Documentos

- [`docs/00-instalacion-windows.md`](docs/00-instalacion-windows.md) — instalar
  LeRobot en Windows 11: ACLs, cachés, versiones de Python. Con los errores
  literales.
- [`docs/01-conceptos.md`](docs/01-conceptos.md) — glosario de robótica e
  imitation learning para gente que viene de software.
