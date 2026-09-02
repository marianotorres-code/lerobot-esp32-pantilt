# 02 — Qué es un LeRobotDataset por dentro

Exploración de tres datasets públicos del hub, elegidos con perfiles
deliberadamente distintos para ver qué cambia y qué se mantiene.

Reproducible con:

```powershell
. D:\robotics-lab\env.ps1
& $PY scripts\explore_dataset.py lerobot/pusht
```

## Los tres datasets

| | `lerobot/pusht` | `lerobot/svla_so101_pickplace` | `lerobot/aloha_sim_insertion_human` |
|---|---|---|---|
| Origen | simulación 2D | **robot real** SO-101 | simulación bimanual |
| Tarea | empujar una pieza en T a su sitio | meter un lego rosa en una caja | insertar un pivote en un casquillo |
| Demostraciones | 206 | 50 | 50 |
| Frames | 25.650 | 11.939 | 25.000 |
| fps | 10 | 30 | 50 |
| Duración | 42,8 min | 6,6 min | 8,3 min |
| Grados de libertad | 2 | 6 | 14 (dos brazos de 7) |
| Cámaras | 1 × 96×96 | 2 × 480×640 | 1 × 480×640 |
| Tamaño en disco | 7,4 MB | 82,1 MB | ~250 MB |

Tres cosas que saltan a la vista:

**El número de demostraciones es pequeño.** 50 episodios. En visión por
computador eso no es ni un dataset de juguete. Aquí es lo normal, y es la razón
de ser del campo: cada demostración cuesta que un humano teleopere un robot en
tiempo real. `pusht` puede permitirse 206 porque es simulación.

**Los grados de libertad varían muchísimo** — de 2 a 14 — pero el formato es
idéntico. Esa es la gracia: la misma arquitectura de política se entrena sobre
cualquiera de ellos cambiando solo la dimensión de entrada y salida.

**`pusht` tiene 2 grados de libertad, igual que mi pan/tilt.** Es el que uso
para entrenar en la Parte A precisamente por eso: lo que aprenda ahí se traduce
casi directo a la Parte B.

## Qué contiene un frame

```
observation.image      video     (96, 96, 3)     <- lo que ve
observation.state      float32   (2,)            <- dónde está
action                 float32   (2,)            <- qué hizo el humano
episode_index          int64     (1,)            <- contabilidad
frame_index            int64     (1,)
timestamp              float32   (1,)
next.reward            float32   (1,)            <- solo algunos datasets
next.done              bool      (1,)
next.success           bool      (1,)
index                  int64     (1,)
task_index             int64     (1,)
```

Las tres primeras son los datos de verdad; el resto es contabilidad que LeRobot
añade a todos los datasets para poder indexar, ordenar y cortar por episodios.

`next.reward` / `next.success` solo aparecen en datasets que vienen de un
entorno con recompensa (`pusht`, `xarm`). Los de robot real no las traen: nadie
está puntuando automáticamente si el lego entró en la caja. Es un recordatorio
de que **imitation learning no necesita recompensa** — esa es justo su ventaja
frente a RL.

### Lo que devuelve `dataset[0]`

Un `LeRobotDataset` es un `torch.utils.data.Dataset` corriente: lo indexas y te
da un diccionario de tensores.

```
observation.image      torch.float32   (3, 96, 96)    [0.235, 1.000]
observation.state      torch.float32   (2,)           [97.000, 222.000]
action                 torch.float32   (2,)           [71.000, 233.000]
task                   str             -              'Push the T-shaped block...'
```

Dos detalles que se pasan por alto y luego duelen:

1. **La imagen sale en CHW, no en HWC.** OpenCV te da `(alto, ancho, canal)`;
   PyTorch quiere `(canal, alto, ancho)`. LeRobot ya hace la conversión al
   decodificar el vídeo. Si la repites, entrenas con basura.
2. **La imagen ya viene en `float32` normalizada a [0,1].** No la vuelvas a
   dividir por 255.

## `state` vs `action`: la relación clave

Medí la diferencia media entre ambos, como fracción del rango de cada dimensión:

| Dataset | `\|action − state\|` medio | % del rango |
|---|---|---|
| `pusht` | 12,68 | 2,9 % |
| `svla_so101_pickplace` | 2,90 | 3,1 % |
| `aloha_sim_insertion_human` | 0,053 | 4,5 % |

En los tres, `action` y `state` viven **en el mismo espacio y con valores casi
idénticos**. Esto confirma lo del glosario: la acción es una *posición objetivo*
de las articulaciones, no una velocidad ni un par. La diferencia entre ambos es
simplemente el error de seguimiento del controlador del motor.

Es una propiedad importante y contraintuitiva. Cuando abres tu primer dataset y
ves dos columnas casi iguales, parece que algo está duplicado. No lo está.

> **Un error mío que dejo escrito.** La primera versión de este análisis
> normalizaba por la media absoluta del estado. Con `aloha` daba 15,6 % y
> concluía "difieren bastante", cuando mirando dimensión a dimensión eran casi
> idénticos. El motivo: las articulaciones de aloha van en radianes centrados en
> cero, así que la media absoluta es diminuta y cualquier diferencia parece
> enorme. Normalizar por el **rango** de cada dimensión lo arregla, porque no
> depende de dónde esté el cero. Métrica mal elegida, conclusión equivocada.

## Longitud de los episodios

En `pusht`: mínimo 49 frames, media 125, máximo 246. Un factor 5 entre el más
corto y el más largo.

Esto no es ruido, es información. Las demostraciones largas son intentos en los
que la pieza empezó lejos o se atascó. Un modelo entrenado con esta mezcla tiene
que aprender ambos comportamientos. Para la Parte B es un aviso: si mis 50 demos
del pan/tilt tienen longitudes muy dispares, o la tarea es de dificultad
variable, o estoy teleoperando de forma inconsistente — y lo segundo es un
problema, no un dato.

## El formato en disco (v3.0)

**Ojo con la documentación que encuentres por ahí: casi toda describe el formato
v2.1.** La versión actual es la **v3.0** (`CODEBASE_VERSION = "v3.0"` en
`lerobot/datasets/dataset_metadata.py`) y la diferencia es estructural.

`lerobot/pusht` completo:

```
meta/   4 ficheros   112,8 KB
  meta/episodes/chunk-000/file-000.parquet   104,1 KB
  meta/info.json                               2,2 KB
  meta/stats.json                              4,3 KB
  meta/tasks.parquet                           2,2 KB
data/   1 fichero    658,6 KB
  data/chunk-000/file-000.parquet            658,6 KB
videos/ 1 fichero      6,6 MB
  videos/observation.image/chunk-000/file-000.mp4   6,6 MB
```

| Ruta | Qué guarda |
|---|---|
| `meta/info.json` | fps, features, plantillas de ruta, totales |
| `meta/stats.json` | media / desviación / min / max para normalizar |
| `meta/tasks.parquet` | las instrucciones en lenguaje natural |
| `meta/episodes/` | índice de episodios: dónde vive cada uno |
| `data/chunk-XXX/` | todo lo NO visual, en parquet |
| `videos/<cámara>/` | las imágenes, codificadas en mp4 |

### Decisión de diseño 1: las imágenes van en vídeo, no en el parquet

Míralo en `svla_so101_pickplace`, que es el caso extremo:

```
data/    361,3 KB     <- posiciones de las 6 articulaciones, 11.939 frames
videos/   81,7 MB     <- las dos cámaras
```

Los datos numéricos son **el 0,4 %** del dataset. Todo el peso son imágenes.
Guardarlas como PNG sueltos multiplicaría el tamaño por un orden de magnitud.

El coste de esta decisión es que hay que **decodificar vídeo al vuelo** en cada
acceso. Por eso `torchcodec` no es un adorno sino una dependencia crítica, y por
eso su fallo de instalación en Windows (ver
[`docs/00`](00-instalacion-windows.md), Problema 7) bloquea todo.

### Decisión de diseño 2: muchos episodios por fichero

Esto es lo que cambió de v2.1 a v3.0. En v2.1 había un parquet y un mp4 **por
episodio**. Un dataset de 1.000 episodios con 3 cámaras eran 4.000 ficheros.

En v3.0 se agrupan en ficheros grandes (~100 MB de datos, ~200 MB de vídeo) y
cada episodio es un **rango de tiempo dentro de un mp4 compartido**:

```
 episodio  fichero  desde (s)  hasta (s)
        0        0        0.0       16.1
        1        0       16.1       27.9
        2        0       27.9       42.0
        3        0       42.0       57.9
        4        0       57.9       73.8
```

Los 206 episodios de `pusht` viven en **un solo mp4**. Por eso el decodificador
tiene que saber buscar por timestamp, no solo abrir ficheros.

### La tabla `meta.episodes`

```
episode_index
data/chunk_index
data/file_index
dataset_from_index                              <- primer frame en el índice global
dataset_to_index                                <- último
videos/observation.image/chunk_index
videos/observation.image/file_index
videos/observation.image/from_timestamp         <- dónde empieza en el mp4
videos/observation.image/to_timestamp
tasks
length
```

> **Nota de API.** Casi todos los tutoriales acceden a esto con
> `dataset.episode_data_index["from"]`. **Ese atributo ya no existe** en lerobot
> 0.6.2 y da `AttributeError`. La información está ahora en
> `dataset.meta.episodes`, que es un `datasets.Dataset` de Hugging Face.

## Ventanas temporales: cómo se alimenta a ACT

ACT y Diffusion Policy no consumen un frame suelto — necesitan una secuencia de
acciones futuras (el *action chunking* del glosario). Eso se pide con
`delta_timestamps`:

```python
fps = 10
horizonte = 16
delta = {"action": [i / fps for i in range(horizonte)]}
ds = LeRobotDataset("lerobot/pusht", delta_timestamps=delta)
```

```
sin delta_timestamps:  action shape = (2,)
con horizonte de 16:   action shape = (16, 2)
```

Ahora cada elemento trae las 16 acciones siguientes. **Aquí es donde el action
chunking deja de ser un concepto de paper y se convierte en una línea de
código**: el dataset entrega bloques, y el modelo aprende a predecir bloques.

Al final de un episodio no hay 16 acciones futuras. LeRobot rellena y marca el
relleno en `action_is_pad`, para que la función de pérdida lo ignore:

```
action_is_pad: [False, False, ..., True, True]
```

Sin esa máscara, el modelo aprendería a predecir el relleno como si fuera una
acción real — un error silencioso y bastante dañino.

## Qué me llevo a la Parte B

De todo esto, lo que condiciona el diseño de mi propio dataset:

1. **Grabar `action` como posición objetivo**, y `observation.state` como
   posición real leída del servo. No dos copias del mismo número — por eso el
   firmware interpola hacia el objetivo en vez de saltar (ver el slew rate
   limiter en el `.ino`).
2. **Vídeo, no PNG.** `use_videos=True` en `LeRobotDataset.create`.
3. **Consistencia entre demostraciones.** La dispersión de longitudes de
   `pusht` es aceptable porque la dificultad varía sola; la mía sería culpa mía.
4. **50 demostraciones es una cantidad normal**, no un atajo.
