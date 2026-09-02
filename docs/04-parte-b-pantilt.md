# 04 — Mi propio robot: pan/tilt con ESP32, 2 servos y una webcam

Parte B. El objetivo no es tener un robot útil, es **cerrar el bucle completo**
con hardware propio: teleoperar, grabar demostraciones en el mismo formato que
usa LeRobot, y entrenar una política sobre ellas.

> **Estado honesto.** El firmware y todo el software Python están escritos y
> revisados, pero **no probados sobre hardware real** — no tengo la plataforma
> montada mientras escribo esto. Todo lo que aparece aquí marcado como
> *verificado* lo he ejecutado; el resto está marcado como *sin probar*. No voy
> a fingir resultados que no tengo.
>
> Lo que **sí** está verificado, y es la afirmación central del punto 5, es que
> el formato que produce el grabador es un `LeRobotDataset` válido y entrenable.
> Ver [Verificación sin hardware](#verificación-sin-hardware) al final.

## Por qué un pan/tilt y no un brazo

Dos servos son 2 grados de libertad. Eso es exactamente lo mismo que
`lerobot/pusht`, el dataset con el que entreno en la Parte A. El espacio de
acciones es un vector de dos números que cabe en una gráfica, y cuando algo
falle voy a poder mirarlo directamente en vez de adivinar entre 14 dimensiones.

Un pan/tilt no puede coger nada, así que la tarea no puede ser manipulación. La
tarea es **centrar un objeto de color en el frame**: muevo la cámara hasta que
el objeto queda en el medio. Es la tarea más simple que sigue siendo un problema
de control visuomotor de verdad — la política tiene que mirar la imagen para
decidir hacia dónde moverse.

## Montaje

### Componentes

- ESP32 (cualquier placa de desarrollo con USB)
- 2 servos, tipo SG90 o MG90S
- Soporte pan/tilt (o cinta aislante y paciencia)
- Webcam USB, montada sobre el tilt
- **Fuente externa de 5 V** — no opcional, ver abajo

### Cableado

| Servo | GPIO |
|---|---|
| pan (giro horizontal) | 13 |
| tilt (inclinación vertical) | 14 |

Evito el GPIO 12 aunque parezca libre: es un pin de *strapping* (MTDI). Si el
servo lo deja en alto durante el arranque, el ESP32 configura el voltaje de la
flash a 1,8 V y **no bootea**. Es un fallo desconcertante porque la placa
simplemente no da señales de vida.

### Alimentación — el fallo número uno

**Los servos no se alimentan del pin 5V del ESP32.** Un SG90 pide picos de unos
700 mA al arrancar el movimiento; el regulador de la placa no da eso. El
síntoma es que la placa se reinicia sola justo cuando mueves los dos servos a la
vez (*brownout*), y como se reinicia también se reabre el puerto serie, parece
un problema de software.

```
Fuente externa 5 V  ──►  V+ de los dos servos
GND de la fuente    ──►  GND del ESP32     ← masa común, imprescindible
Señal de cada servo ──►  su GPIO
```

Sin la masa común no hay referencia de tensión y los servos se mueven de forma
errática. Es el error clásico con servos y no da ningún mensaje de error.

## Firmware

[`firmware/esp32_pantilt/esp32_pantilt.ino`](../firmware/esp32_pantilt/esp32_pantilt.ino)

Dependencia: librería **ESP32Servo** de Kevin Harrington, desde el Gestor de
Librerías del IDE de Arduino. La `Servo.h` estándar de Arduino **no** funciona
en ESP32.

### La decisión de diseño importante: el slew rate limiter

El firmware **no salta** a la posición objetivo. Interpola hacia ella a una
velocidad máxima (180 °/s por defecto), en un bucle de control a 200 Hz.

Esto no es por suavidad estética. Es lo que hace que el dataset tenga algo que
aprender:

```
action            = posición OBJETIVO que mandó el host   (target)
observation.state = posición REAL donde está el servo     (current)
```

Si el servo saltara al objetivo instantáneamente, `state` y `action` serían el
mismo número y el dataset no contendría dinámica ninguna. Con el limitador,
`state` persigue a `action` con un retardo físico, igual que pasa en un brazo de
verdad — y como vimos en [`docs/02`](02-datasets-lerobot.md), esa pequeña
diferencia entre ambos es exactamente lo que caracteriza a los datasets reales
de LeRobot.

### Protocolo

ASCII por línea, 115200 baudios. Cada comando devuelve **exactamente una
línea**, lo que permite hacer petición/respuesta síncrona sin ambigüedad.

| Comando | Efecto |
|---|---|
| `M <pan> <tilt>` | fija la posición objetivo en grados |
| `S` | pide el estado |
| `H` | vuelve al centro (90, 90) |
| `V <°/s>` | cambia la velocidad de slew |
| `E <0\|1>` | suelta / activa los servos |
| `P` | ping, responde `PONG` |

Respuesta de estado:

```
S <pan> <tilt> <tgt_pan> <tgt_tilt> <millis>
```

No hay streaming automático, a propósito: **el host marca el ritmo de muestreo**.
Así el timestamp de cada frame del dataset es el del host, que es el mismo reloj
que timestampea la webcam. Si la placa emitiera por su cuenta, tendría dos
relojes que sincronizar.

Los límites mecánicos se aplican **en el firmware**, no solo en Python. El
firmware es la última línea de defensa del hardware y no debe fiarse de quien le
manda comandos. Ajusta `PAN_MIN`/`PAN_MAX`/`TILT_MIN`/`TILT_MAX` a tu montaje
**antes** de mover nada: si el soporte choca, el servo sigue empujando y se
carga el engranaje.

## Cliente Python

[`python/lerobot_pantilt/pantilt.py`](../python/lerobot_pantilt/pantilt.py)

```python
from lerobot_pantilt.pantilt import PanTilt

with PanTilt("COM5") as robot:
    robot.home()
    estado = robot.move(100.0, 85.0)
    print(estado.state_vector, estado.action_vector)
```

### Trampa: abrir el puerto serie reinicia el ESP32

Al abrir el puerto, el driver activa las líneas DTR y RTS, que en las placas de
desarrollo están cableadas al circuito de auto-reset (EN y GPIO0). O sea que
**abrir el puerto reinicia la placa**. No es un bug: es el mecanismo que usa el
IDE de Arduino para entrar en modo flash.

La consecuencia es que no puedes abrir el puerto y mandar un comando
inmediatamente — se pierde, porque la placa está en el bootloader. Por eso
`connect()` espera a la línea `READY` antes de dar por buena la conexión.

Otro clásico: **solo un proceso puede tener abierto un puerto COM**. Si tienes
el Monitor Serie del IDE de Arduino abierto, Python no podrá conectar.

## Grabación de demostraciones

[`python/record_pantilt.py`](../python/record_pantilt.py)

```powershell
. D:\robotics-lab\env.ps1
& $PY python\record_pantilt.py --repo-id maria/pantilt-centrar --port COM5
```

| Tecla | Acción |
|---|---|
| flechas / WASD | mover pan y tilt |
| ESPACIO | guardar el episodio y preparar el siguiente |
| R | descartar el episodio y repetirlo |
| H | volver al centro |
| Q / ESC | terminar |

Escribe directamente en formato `LeRobotDataset` v3.0, así que el dataset
resultante se entrena con `lerobot-train` sin ninguna conversión — exactamente
igual que uno del hub.

### El orden del bucle importa

```
1. capturar imagen de la webcam
2. leer el estado real del servo        ← observación en el instante t
3. leer el teclado → nuevo objetivo     ← decisión humana
4. mandar el objetivo a la placa
5. guardar el frame (imagen, state, action)
6. dormir hasta el siguiente tick
```

El frame que se guarda es "estaba **aquí**, y le pedí ir **allá**". Si leyeras
el estado *después* de mandar el movimiento, estarías grabando el efecto de la
acción como si fuera su causa, y la política aprendería a predecir el pasado.

### Cuatro cosas que van a estropear el dataset en silencio

Ninguna de estas da error. Todas producen un dataset que entrena y no funciona.

1. **BGR vs RGB.** OpenCV entrega BGR; LeRobot espera RGB. Sin
   `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` el modelo entrena igual, pero los
   vídeos salen azules y el dataset es incompatible con cualquier otro.
2. **El buffer de la webcam.** Sin `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`, OpenCV
   acumula frames y acabas emparejando una imagen de hace medio segundo con la
   posición actual del servo. El dataset queda desincronizado y no hay nada que
   aprender.
3. **El HUD quemado en la imagen.** El texto de depuración se dibuja sobre una
   **copia** del frame. Si lo pintaras sobre el que guardas, la política
   aprendería a leer el contador de frames en vez de a mirar el objeto — y
   funcionaría de maravilla en validación.
4. **Deriva del reloj.** Usar `sleep(1/30)` acumula el error de cada iteración y
   acabas grabando a 27 Hz creyendo que grabas a 30. Los timestamps del dataset
   mentirían. Por eso el bucle mantiene un reloj absoluto (`next_tick += period`)
   en vez de dormir un intervalo fijo.

## Protocolo de grabación de las 50-100 demos

Esto es lo que más va a determinar si la política aprende algo, más que
cualquier hiperparámetro. De [`docs/01`](01-conceptos.md): en imitation
learning la **consistencia** de las demostraciones importa más que la cantidad.

1. **Fija la escena.** Misma iluminación, misma mesa, misma cámara. La política
   va a agarrarse a cualquier pista que correlacione con la respuesta correcta,
   incluida la sombra de la ventana a las 6 de la tarde.
2. **Varía solo lo que quieres que generalice.** Mueve el objeto a una posición
   inicial distinta en cada episodio, cubriendo todo el campo de visión. Si
   siempre empieza a la derecha, la política aprenderá "gira a la izquierda"
   como constante.
3. **Sé consistente en la estrategia.** Siempre la misma forma de acercarte:
   primero pan y luego tilt, o los dos a la vez, pero no unas veces una cosa y
   otras otra. Estrategias mezcladas para observaciones parecidas es
   exactamente el problema de multimodalidad del glosario, y el modelo aprenderá
   el promedio de las dos: no moverse.
4. **Termina igual.** Considera el episodio acabado cuando el objeto lleve ~1
   segundo centrado, no en el instante en que cruza el centro.
5. **Descarta los intentos malos con `R`.** Una demostración en la que te
   equivocaste y corregiste enseña a equivocarse y corregir. A veces eso se
   quiere (ayuda contra el *compounding error*); al principio, no.
6. **Empieza el episodio siempre desde el centro.** El script hace `home()`
   automáticamente al guardar o descartar, así que esto sale gratis.

Con 100 episodios de ~10 segundos a 30 Hz salen unos 30.000 frames, el mismo
orden de magnitud que `lerobot/pusht`.

## Verificación sin hardware

No tengo la plataforma montada, pero la afirmación importante del punto 5 —
*"usa el mismo formato de LeRobot para que sea compatible"* — sí se puede
comprobar. Hay dos pruebas, y son dos cosas distintas.

### 1. ¿Se puede *leer*?

[`scripts/test_dataset_roundtrip.py`](../scripts/test_dataset_roundtrip.py)

Simula el pan/tilt (reproduciendo el slew rate limiter del firmware) y la
webcam, graba 3 episodios con el mismo código de features y el mismo orden de
bucle que el grabador real, y después reabre el resultado con `LeRobotDataset`.

```
  [OK ] episodios                              3  (esperado 3)
  [OK ] frames                                 60  (esperado 60)
  [OK ] fps                                    30  (esperado 30)
  [OK ] robot_type                             esp32_pantilt
  [OK ] shape de la imagen                     (3, 120, 160)
  [OK ] shape de observation.state             (2,)
  [OK ] shape de action                        (2,)
  [OK ] imagen normalizada a [0,1]             [0.000, 1.000]
  [OK ] state va por detras de action          |action-state| medio = 1.9012
  [OK ] chunk de acciones para ACT             (8, 2)
```

La estructura en disco generada es idéntica a la de los datasets públicos:

```
data/chunk-000/file-000.parquet                            8.4 KB
meta/episodes/chunk-000/file-000.parquet                  43.3 KB
meta/info.json                                             2.5 KB
meta/stats.json                                            7.9 KB
meta/tasks.parquet                                         2.1 KB
videos/observation.images.webcam/chunk-000/file-000.mp4    10.0 KB
```

La comprobación que más me importaba es la penúltima: `state` va **por detrás**
de `action`. Si el firmware saltara al objetivo, ambos serían el mismo número y
el dataset no tendría dinámica que aprender, pero la prueba pasaría igualmente.

### 2. ¿Se puede *entrenar*?

[`scripts/test_train_on_own_dataset.py`](../scripts/test_train_on_own_dataset.py)

Que un dataset se lea no significa que se pueda entrenar. El lector es
tolerante; el entrenador no: exige que las features estén clasificadas
correctamente como estado / visual / acción, que existan las estadísticas de
normalización, y que el `chunk_size` encaje con la longitud de los episodios.

Esta prueba genera 8 episodios de 40 frames y lanza `lerobot-train` de verdad
(4 pasos, en CPU para no competir con ningún entrenamiento real):

```
  [OK ] lerobot-train sale con codigo 0
  [OK ] checkpoint escrito en pretrained_model
        contenido: config.json, model.safetensors,
                   policy_postprocessor.json,
                   policy_postprocessor_step_0_unnormalizer_processor.safetensors,
                   policy_preprocessor.json,
                   policy_preprocessor_step_3_normalizer_processor.safetensors,
                   train_config.json
```

Que aparezcan los `normalizer_processor` es la señal de que LeRobot reconoció
las features y calculó la normalización a partir de mi `meta/stats.json`. Eso es
lo que separa "escribí un dict con las claves correctas" de "esto es un dataset
de LeRobot".

**Lo que estas pruebas NO demuestran:** que el firmware funcione en una placa
real, que los servos se muevan, que la webcam sincronice bien con el serial, ni
que la política aprenda algo. Con datos sintéticos aleatorios no hay nada que
aprender — la prueba es de formato, no de aprendizaje.

## Entrenar sobre el dataset propio

Sin probar todavía — hasta que no existan las demos no hay nada que entrenar.
El comando debería ser el mismo que en la Parte A cambiando el `repo_id`:

```powershell
& $PY -m lerobot.scripts.lerobot_train `
  --dataset.repo_id=maria/pantilt-centrar `
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda `
  --output_dir=D:\robotics-lab\outputs\act_pantilt `
  --steps=20000 --batch_size=32 --wandb.enable=false
```

Sin `--env.type`, porque no hay simulador de mi pan/tilt. Eso significa que **no
se puede medir la tasa de éxito automáticamente**: la única evaluación posible
es ejecutar la política en el robot real y mirar. Es una diferencia importante
respecto a la Parte A y conviene tenerla clara antes de empezar.
