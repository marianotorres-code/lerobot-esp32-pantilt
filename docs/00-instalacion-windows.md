# 00 — Instalar LeRobot en Windows 11: lo que falla y cómo se arregla

> Bitácora honesta. Cada problema aquí es uno que me encontré de verdad en esta
> máquina, con el error literal y la solución que funcionó. No es una guía
> idealizada.

## La máquina

| Componente | Valor |
|---|---|
| SO | Windows 11 Pro 10.0.26200 |
| GPU | NVIDIA GeForce RTX 3050, 8 GB VRAM |
| Driver NVIDIA | 591.86 (soporta hasta CUDA 13.1) |
| Python | 3.12.10 (y 3.14 como default del sistema) |
| Shell | PowerShell 5.1 + Git Bash |
| Disco C: | 223 GB — **3.0 GB libres (99% lleno)** |
| Disco D: | 954 GB — 280 GB libres |
| Compilador MSVC | ausente (`cl.exe` no está en PATH) |
| conda | no instalado |

## Problema 1 — El disco de sistema no tiene espacio

**Síntoma anticipado.** Antes de instalar nada:

```
C:  223G  220G  3.0G  99% /c
```

PyTorch con CUDA ocupa ~2.5 GB solo el wheel. Sumando `lerobot`, sus
dependencias y dos o tres datasets del hub, esto no cabe ni de lejos. Si hubiera
lanzado `pip install` a ciegas, habría reventado a mitad de la descarga y encima
habría dejado el sistema sin espacio para el archivo de paginación.

**Solución.** Separar código de datos:

- Repo (código + docs, pocos MB) → `C:\Users\maria\ESP32`
- Venv, cachés, datasets, checkpoints → `D:\robotics-lab`

Esto no basta con crear la carpeta: hay tres cachés que por defecto escriben en
C: y hay que redirigir explícitamente. Ver Problema 3.

## Problema 2 — No se puede crear una carpeta en la raíz de D:

```
$ mkdir /d/robotics-lab
mkdir: cannot create directory 'D:\robotics-lab': Permission denied
```

Al principio pensé que era el sandbox del agente. No lo era. `icacls D:\`:

```
D:\ BUILTIN\Usuarios:(OI)(CI)(RX)
    NT AUTHORITY\Authenticated Users:(OI)(CI)(RX)
    BUILTIN\Administradores:(OI)(CI)(F)
    NT AUTHORITY\SYSTEM:(OI)(CI)(F)
```

`RX` = read + execute. Sin `W`. La raíz de un disco secundario en Windows solo
deja escribir a administradores. Y aunque mi usuario **está** en el grupo
Administradores, el token de la sesión lo marca como *"grupo usado solo para
denegar"* — es el token filtrado de UAC. Ser admin no sirve si el proceso no
está elevado.

**Trampa importante:** crear la carpeta con UAC **no es suficiente**. La carpeta
nueva hereda la ACL del padre, o sea `RX`, así que sigue siendo de solo lectura
para procesos no elevados:

```
D:\robotics-lab BUILTIN\Usuarios:(I)(OI)(CI)(RX)      <- heredado, sin W
WRITE: DENIED - Acceso denegado a la ruta 'D:\robotics-lab\.__wtest'
```

**Solución.** Crear la carpeta *y* concederse control total explícito, una sola
vez, elevado:

```powershell
# En una PowerShell como administrador
New-Item -ItemType Directory "D:\robotics-lab"
icacls "D:\robotics-lab" /grant "$env:USERNAME:(OI)(CI)F"
```

`(OI)(CI)` = Object Inherit + Container Inherit, o sea que archivos y
subcarpetas futuros también lo heredan. `F` = Full control. Después de esto todo
el trabajo se hace **sin elevar**, que es como debe ser.

Verificación:

```
D:\robotics-lab BRUTISHPC\maria:(OI)(CI)(F)   <- ACE explícito
WRITE: OK
```

## Problema 3 — Las cachés por defecto apuntan a C:

Aunque el venv esté en D:, estas tres siguen escribiendo en C: y te llenan el
disco igual:

| Variable | Default (C:) | Qué guarda |
|---|---|---|
| `HF_HOME` | `C:\Users\maria\.cache\huggingface` | datasets y modelos del hub |
| `PIP_CACHE_DIR` | `C:\Users\maria\AppData\Local\pip` | wheels descargados |
| `TORCH_HOME` | `C:\Users\maria\.cache\torch` | pesos preentrenados |
| `TMP` / `TEMP` | `C:\Users\maria\AppData\Local\Temp` | **descompresión de wheels** |

La cuarta es la que más duele y la que nadie menciona: **pip descomprime el
wheel en `%TEMP%` antes de instalarlo**. Con torch+cu128 (~2.5 GB) eso solo ya
no cabe en 3 GB libres, y el error que sale (`No space left on device` a mitad
de un `pip install` de 20 minutos) no te dice que el culpable es TEMP.

**Solución.** `D:\robotics-lab\env.ps1`, que se hace dot-source antes de
cualquier comando:

```powershell
. D:\robotics-lab\env.ps1
```

## Problema 4 — Espacios en la ruta del venv

Mi primer intento fue `D:\Proyectos python\lerobot-lab\venv` (era la única
carpeta escribible que encontré antes de resolver el Problema 2). Un venv en una
ruta con espacios funciona *casi* siempre, pero rompe scripts que no citan bien
las rutas, y no merece la pena el riesgo. `D:\robotics-lab` no tiene espacios.

**Nota aparte:** un venv **no se puede mover** de sitio. Guarda rutas absolutas
en `pyvenv.cfg` y en los scripts de `Scripts\`. Si te equivocas de ubicación,
bórralo y recréalo — no lo muevas.

## Problema 5 — La versión de Python

`lerobot` 0.6.2 declara `requires-python = ">=3.12"`. En esta máquina el
`python` del PATH es **3.14**, que es demasiado nuevo: muchas dependencias
científicas todavía no publican wheels para 3.14, y sin `cl.exe` instalado no
hay forma de compilarlas desde fuente (Problema 6).

**Solución.** Crear el venv apuntando explícitamente al 3.12:

```powershell
& "C:\Users\maria\AppData\Local\Programs\Python\Python312\python.exe" -m venv "D:\robotics-lab\venv"
```

Localizar los intérpretes disponibles: `py -0p`

## Problema 6 — No hay compilador C++ (resultó no importar)

`cl.exe` no está en el PATH, o sea que no hay Visual Studio Build Tools.
Esperaba que esto rompiera algo, y anoto el resultado porque es un hallazgo
útil aunque sea negativo: **no hizo falta**. Todas las dependencias de
`lerobot[dataset,training,pusht,diffusion,hardware,viz]` tienen wheel
precompilado para Windows + Python 3.12. Cero compilaciones desde fuente.

Esto incluye las que daba por perdidas: `pymunk` (motor de física de
`gym-pusht`), `av` (bindings de FFmpeg), `scikit-image` y `torchcodec`.

Comandos que funcionaron tal cual:

```powershell
. D:\robotics-lab\env.ps1
& $PY -m pip install "torch==2.11.*" torchvision --index-url https://download.pytorch.org/whl/cu128
cd D:\robotics-lab\lerobot
& $PY -m pip install -e ".[dataset,training,pusht,diffusion]"
& $PY -m pip install -e ".[hardware,viz]"
```

Verificación de CUDA:

```
torch 2.11.0+cu128   torchvision 0.26.0+cu128
cuda_available True  cuda_build 12.8
device NVIDIA GeForce RTX 3050
```

## Problema 7 — torchcodec no carga: FFmpeg y las DLLs de Windows

Este es **el fallo serio de Windows** de todo el proceso, y el único que costó
de verdad. `torchcodec` es el decodificador de vídeo con el que LeRobot lee los
mp4 de los datasets: sin él no se puede abrir ni un solo dataset del hub.

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment...
     On Windows, ensure you've installed the "full-shared" version which ships DLLs.
```

`torchcodec` trae sus propias DLLs (`libtorchcodec_core4.dll` … `core8.dll`,
una por versión mayor de FFmpeg), pero estas dependen de las DLLs de FFmpeg
(`avcodec`, `avformat`, `avutil`, `swscale`, `swresample`), que **Windows no
trae**. En Linux las instala el gestor de paquetes y por eso nadie documenta
este paso.

### Trampa 7a — la descarga obvia de FFmpeg es la incorrecta

El build que todo el mundo enlaza de [BtbN](https://github.com/BtbN/FFmpeg-Builds)
es `ffmpeg-master-latest-win64-gpl-shared.zip`. Lo bajé y trae:

```
avcodec-63.dll   avformat-63.dll   avutil-61.dll
```

`avcodec-63` es FFmpeg **9-dev**. `torchcodec` 0.11.1 solo soporta las mayores
**4 a 8**. O sea que el build "latest" es demasiado nuevo y no sirve, y el
mensaje de error no lo dice en ningún momento.

Hay que coger una release etiquetada. Las que existen ahora mismo en ese repo
son `n8.1` y `n9.0`; la buena es la 8.1:

```powershell
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip"
```

```
ffmpeg version n8.1.2   avcodec-62.dll   avformat-62.dll   avutil-60.dll
```

Segundo detalle: tiene que ser un build **`shared`**. Los `static`, que son los
que más se descargan, meten FFmpeg dentro del `ffmpeg.exe` y no traen DLLs
sueltas, así que no le sirven de nada a torchcodec.

### Trampa 7b — añadir FFmpeg al PATH no funciona

Con FFmpeg 8.1 descomprimido, añadí `D:\robotics-lab\ffmpeg\bin` al `PATH`.
Sigue fallando, ahora con un error peor:

```
FileNotFoundError: Could not find module
'...\site-packages\torchcodec\libtorchcodec_core4.dll' (or one of its dependencies)
```

Dice que no encuentra `libtorchcodec_core4.dll` — que **está ahí**, la puedes
ver con `dir`. La que falta es una de sus *dependencias*, y el mensaje no dice
cuál.

La causa: **desde Python 3.8, en Windows, `PATH` ya no se usa para resolver las
DLLs de las que dependen los módulos de extensión.** Fue un cambio de seguridad
deliberado de CPython, para evitar secuestro de DLLs. Hay que registrar el
directorio explícitamente con
[`os.add_dll_directory()`](https://docs.python.org/3/library/os.html#os.add_dll_directory).

`torchcodec` 0.11.1 no lo hace por su cuenta (lo comprobé leyendo
`_internally_replaced_utils.py`: llama a `torch.ops.load_library()` a pelo).

**Solución.** Un `sitecustomize.py` en el `site-packages` del venv. Python lo
importa solo al arrancar el intérprete, así que el arreglo aplica a todos los
procesos del entorno — `lerobot-train`, notebooks, scripts propios — sin tener
que acordarse en cada script:

```powershell
copy scripts\sitecustomize.py D:\robotics-lab\venv\Lib\site-packages\
```

El fichero está en [`scripts/sitecustomize.py`](../scripts/sitecustomize.py) con
la explicación completa. Después:

```
torchcodec 0.11.1+cpu - VideoDecoder OK
```

**Alternativa si esto te da guerra:** LeRobot lleva un decodificador de respaldo
basado en PyAV (`video_backend="pyav"`), y `av` sí trae FFmpeg dentro del wheel.
Es más lento pero no necesita nada de lo anterior. Preferí arreglar torchcodec
porque es el backend por defecto y el que se usa en el resto de la documentación.

### Nota sobre `torchcodec 0.11.1+cpu`

Se instaló la variante `+cpu` aunque torch sea `+cu128`. Es lo correcto: esa
etiqueta se refiere a decodificación de vídeo por GPU (NVDEC), que es una
optimización aparte. La decodificación por CPU va de sobra para estos datasets.

## Problema 8 — un falso positivo mío

Mi primera prueba de humo dio `ModuleNotFoundError: No module named 'datasets'`.
Estuve a punto de documentarlo como un conflicto del resolvedor. No lo era:
lancé la prueba mientras `pip` todavía estaba escribiendo ese paquete —
`datasets` era de los últimos de la lista de instalación.

Lo dejo escrito porque es el tipo de error que se documenta mal con facilidad.
Antes de dar por bueno un fallo, comprueba que la instalación terminó de verdad
(código de salida, no "parece que ya está").

## Estado final del entorno

Todo verificado ejecutando, no leyendo.

| Componente | Versión |
|---|---|
| lerobot | 0.6.2 (editable, commit `fbb811f`) |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchcodec | 0.11.1+cpu |
| FFmpeg | n8.1.2 (shared) |
| numpy | 2.2.6 |
| opencv | 5.0.0 |
| gymnasium / gym-pusht | 1.3.0 / 0.1.6 |
| diffusers | 0.39.0 |
| datasets | 4.8.5 |
| pyserial | 3.5 |
| CUDA | disponible, RTX 3050 |
| Formato de dataset | v3.0 |

- [x] Venv 3.12 en `D:\robotics-lab\venv`
- [x] Cachés redirigidas a D:
- [x] LeRobot 0.6.2 clonado y en modo editable
- [x] PyTorch + CUDA verificado con un matmul en GPU
- [x] torchcodec cargando (FFmpeg 8.1 + `sitecustomize.py`)

## Resumen para quien venga detrás

Si solo quieres la receta, en orden:

1. Usa Python **3.12** explícitamente (`py -0p` para localizarlo). El 3.13/3.14
   del PATH no vale.
2. Pon el venv y las cachés en un disco con espacio. Redirige `HF_HOME`,
   `PIP_CACHE_DIR`, `TORCH_HOME` y **`TMP`/`TEMP`** (esta última es la que
   revienta la instalación).
3. Instala torch desde el índice de PyTorch **antes** que lerobot, para elegir
   tú la build de CUDA.
4. Descarga FFmpeg **8.x**, build **shared**, no el `master-latest` ni el
   `static`.
5. Registra su carpeta `bin` con `os.add_dll_directory()` vía `sitecustomize.py`.
   Añadirlo al `PATH` no funciona.
6. No necesitas Visual Studio Build Tools.
