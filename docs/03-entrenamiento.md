# 03 — Entrenar ACT y evaluarla en simulación

Punto 3 de la Parte A. El objetivo declarado **no es que funcione bien**, es
entender el pipeline entero: dataset → política → checkpoint → evaluación en un
simulador → métrica.

## Qué entreno y por qué

**Política: ACT** (Action Chunking with Transformers). Empiezo por ella y no por
Diffusion Policy por tres razones prácticas: entrena más rápido, la inferencia
es más rápida (no hay bucle de *denoising*), y tiene menos hiperparámetros que
ajustar mal. La justificación conceptual está en
[`docs/01`](01-conceptos.md#act--action-chunking-with-transformers).

**Dataset: `lerobot/pusht`.** 2 grados de libertad — los mismos que mi pan/tilt
de la Parte B — e imágenes de 96×96, que caben de sobra en una RTX 3050. Y sobre
todo: tiene **simulador propio** (`gym-pusht`), que es lo que permite medir una
tasa de éxito automáticamente. Sin simulador, la única evaluación posible es
mirar el robot y opinar.

## Configuración por defecto de ACT

De `lerobot/policies/act/configuration_act.py`:

| Parámetro | Valor | Qué significa |
|---|---|---|
| `chunk_size` | 100 | cuántas acciones predice de golpe |
| `n_action_steps` | 100 | cuántas ejecuta antes de volver a mirar |
| `n_obs_steps` | 1 | ve **un solo** frame, sin historia |
| `vision_backbone` | `resnet18` | codificador de imagen, uno por cámara |
| `dim_model` | 512 | anchura del transformer |
| `n_encoder_layers` | 4 | |
| `n_decoder_layers` | 1 | |
| `use_vae` | `True` | el objetivo variacional del CVAE |
| `optimizer_lr` | 1e-5 | |
| `temporal_ensemble_coeff` | `None` | ensembling temporal desactivado por defecto |

Dos cosas que sorprenden viniendo de software:

**`n_obs_steps = 1`.** La política ve una única imagen, sin ninguna memoria de
lo anterior. Es tentador pensar que necesita historia — no la necesita, porque
predice 100 acciones de una vez. El horizonte largo está en la *salida*, no en
la entrada.

**`chunk_size == n_action_steps == 100`.** Predice 100 acciones y ejecuta las
100 antes de volver a mirar. A 10 Hz eso son 10 segundos a ciegas. Suena
temerario, y lo es — es justo el compromiso del action chunking: menos
oportunidades de acumular error (el *compounding error* del glosario), pero
también menos capacidad de reaccionar a algo inesperado.

## El comando

```powershell
. D:\robotics-lab\env.ps1
& $PY -m lerobot.scripts.lerobot_train `
  --dataset.repo_id=lerobot/pusht `
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda `
  --env.type=pusht `
  --output_dir=D:\robotics-lab\outputs\act_pusht `
  --job_name=act_pusht `
  --steps=30000 --batch_size=64 --num_workers=4 `
  --log_freq=250 --save_freq=10000 `
  --env_eval_freq=10000 --eval.n_episodes=20 --eval.batch_size=10 --eval.use_async_envs=false `
  --wandb.enable=false
```

## Los cuatro tropiezos hasta que arrancó

Ninguno está en los tutoriales. Los pongo en orden de aparición.

### 1. Intenta subir el modelo al hub por defecto

```
ValueError: 'repo_id' argument missing. Please specify it to push the model to the hub.
```

Falla **antes de empezar a entrenar**, en `cfg.validate()`. Subir a Hugging Face
es el comportamiento por defecto. Se desactiva con `--policy.push_to_hub=false`.

### 2. `--eval_freq` no existe

```
lerobot_train.py: error: unrecognized arguments: --eval_freq=10000
```

El parámetro se llama **`env_eval_freq`**. Es un renombre respecto a lo que
aparece en toda la documentación que circula. Se ve en
`lerobot/configs/train.py`.

### 3. El symlink del checkpoint (Windows)

```
OSError: [WinError 1314] El cliente no dispone de un privilegio requerido:
  '000010' -> 'D:\robotics-lab\outputs\...\checkpoints\last'
```

Este es serio: **el entrenamiento termina y guarda el checkpoint correctamente,
y luego revienta.** `update_last_checkpoint()` en `lerobot/common/train_utils.py`
crea un symlink `checkpoints/last` que apunta al último checkpoint, y Windows no
deja crear symlinks a usuarios sin privilegio.

Como pasa **después** de guardar, es especialmente traicionero: parece que todo
fue bien hasta el crash final. Con `save_freq=10000` reventaría en cada guardado.

No hay ningún flag para desactivarlo. Solución: **activar el Modo Desarrollador**
de Windows, que concede el privilegio de crear symlinks a procesos no elevados.
Una vez, desde una consola elevada:

```
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v AllowDevelopmentWithoutDevLicense /d 1
```

Detalle: después de esto, `New-Item -ItemType SymbolicLink` de PowerShell
**sigue fallando**, pero `os.symlink` de Python funciona. La razón es que CPython
pasa el flag `SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE` a la API de Windows y
PowerShell no. Comprueba con Python, no con PowerShell.

Arregla también los fallos de la caché de Hugging Face (`WinError 1314` al bajar
datasets) y de paso hace que esa caché deduplique con enlaces en vez de duplicar
ficheros, que con el disco justo se agradece.

### 4. `forkserver` no existe en Windows

Al evaluar:

```
ValueError: cannot find context for 'forkserver'
```

`lerobot/envs/configs.py` fija `context = "forkserver"` al construir el
`AsyncVectorEnv` de gymnasium. **`forkserver` solo existe en Unix**; Windows solo
tiene `spawn`. Y `use_async_envs` viene a `True` por defecto.

Solución: `--eval.use_async_envs=false`, que usa `SyncVectorEnv`. El coste es que
los episodios de evaluación se ejecutan en serie y tarda más.

(Nota: el *dataloader* de entrenamiento no tiene este problema — LeRobot ya usa
`dataloader_multiprocessing_context: str | None = "spawn"`. El bug está solo en
los entornos de simulación.)

### 5 (bonus) — perder 53 minutos por poner mal `save_freq`

Este no es un fallo de LeRobot, es un fallo mío, y lo dejo escrito porque es el
tipo de error que se comete una vez.

Lancé la primera corrida con `--steps=30000 --save_freq=10000`. El proceso se
interrumpió en el paso **5.974**, tras 53 minutos. Como el primer guardado no
tocaba hasta el paso 10.000, **no había ningún checkpoint**. Todo perdido.

`save_freq` no es solo "cada cuánto quiero un modelo". Es **cuánto trabajo estoy
dispuesto a perder**. En una corrida larga y desatendida, guardar cada 2.500
pasos cuesta unos megas de disco y convierte una interrupción de una hora en una
de diez minutos.

LeRobot soporta `--resume=true` para continuar desde `checkpoints/last`, pero eso
solo sirve si hay un `last` que reanudar.

Relanzado con `--steps=15000 --save_freq=2500 --env_eval_freq=5000`.

## Cómo leer el log

El log mezcla la barra de progreso de tqdm con las líneas INFO. tqdm reescribe
su línea con retornos de carro (`\r`), así que las entradas INFO quedan pegadas
al final de líneas larguísimas y un `grep` normal engaña: parece que hay 3
puntos de log cuando hay 23.

[`scripts/parse_train_log.py`](../scripts/parse_train_log.py) convierte los `\r`
en saltos de línea antes de parsear:

```powershell
& $PY scripts\parse_train_log.py D:\robotics-lab\logs_act_pusht.txt --every 4
```

Las tres pérdidas que reporta ACT y qué significan:

| Campo | Qué es |
|---|---|
| `l1_loss` | error absoluto entre las acciones predichas y las del humano. **Es la que importa** |
| `kld_loss` | divergencia KL del CVAE: cuánto se aleja el espacio latente de una normal |
| `loss` | la suma ponderada de ambas, que es lo que se minimiza |

Vigilar `loss` a secas puede despistar: al principio baja sobre todo porque cae
la `kld_loss`, no porque el modelo prediga mejor las acciones.

## Rendimiento en una RTX 3050

| | |
|---|---|
| GPU | RTX 3050, 8 GB |
| `batch_size` | 64 |
| VRAM usada | ~5,3 GB |
| Utilización de GPU | ~32 % |
| Velocidad | ~2,6 pasos/s |
| 30.000 pasos | ~3 h |

**La GPU está al 32 %, o sea que el cuello de botella no es la GPU.** Es la
carga de datos: hay que decodificar vídeo mp4 al vuelo para cada frame del batch
(ver [`docs/02`](02-datasets-lerobot.md#decisión-de-diseño-1-las-imágenes-van-en-vídeo-no-en-el-parquet)).
Subir el `batch_size` no ayudaría; lo que ayudaría es más `num_workers`, o
guardar el dataset como imágenes en vez de vídeo a costa de mucho disco.

Es un resultado útil por sí mismo: en imitation learning con vídeo, la GPU no
suele ser el recurso escaso.

## Evaluación

```powershell
& $PY -m lerobot.scripts.lerobot_eval `
  --policy.path=D:\robotics-lab\outputs\act_pusht\checkpoints\last\pretrained_model `
  --env.type=pusht `
  --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false `
  --policy.device=cuda
```

Devuelve:

| Métrica | Qué es |
|---|---|
| `pc_success` | **porcentaje de episodios resueltos.** La métrica que importa |
| `avg_sum_reward` | recompensa acumulada media |
| `avg_max_reward` | mejor recompensa instantánea del episodio |
| `video_paths` | vídeos de los rollouts, para mirar qué hace |

Los vídeos son la parte más informativa y la que más se ignora. Una tasa de
éxito del 20 % no dice **por qué** falla; ver diez rollouts te dice enseguida si
se queda quieto, si oscila, o si se pasa de largo.

### Comprobación temprana del pipeline

Antes de dejar corriendo 3 horas, entrené un modelo de **10 pasos** y lo evalué:

```
{'avg_sum_reward': 2.42, 'avg_max_reward': 0.008, 'pc_success': 0.0, 'n_episodes': 2}
```

0 % de éxito, exactamente lo esperado de un modelo sin entrenar. Pero el pipeline
completo funcionó de punta a punta. Los tropiezos 1, 3 y 4 aparecieron aquí, en
90 segundos, en vez de tras tres horas de entrenamiento.

Es la lección más transferible de toda la Parte A: **verifica el camino completo
con la configuración más pequeña posible antes de gastar tiempo de cómputo.**

## Resultados

*(pendiente — el entrenamiento está corriendo mientras escribo esto)*

| Pasos | pérdida | `pc_success` |
|---|---|---|
| 5 | 64,73 | — |
| 500 | 1,95 | — |
| 10.000 | | |
| 20.000 | | |
| 30.000 | | |

### Expectativa honesta

30.000 pasos de ACT sobre `pusht` **no** van a dar un resultado bueno. Como
referencia, los resultados publicados de LeRobot en pusht usan Diffusion Policy
con del orden de 200.000 pasos. Lo que espero ver es que la pérdida baje de
forma limpia y que `pc_success` pase de 0 a algo distinto de 0. Eso ya demuestra
que el pipeline aprende.

Si la tasa de éxito se queda en 0 %, las hipótesis por orden de probabilidad
son: pasos insuficientes; `chunk_size=100` a 10 Hz es demasiado ciego para
pusht; o ACT simplemente no es la política adecuada para esta tarea —
`pusht` tiene fama de ser **multimodal** (hay muchas formas válidas de empujar
la pieza), que es exactamente el caso en el que Diffusion Policy gana y una
política que minimiza el error cuadrático medio promedia soluciones
incompatibles.
