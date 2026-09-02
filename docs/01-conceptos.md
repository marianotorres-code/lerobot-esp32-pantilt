# 01 — Conceptos de robótica e imitation learning, para gente que viene de software

> Escrito conforme los voy usando. Si un concepto no aparece aquí es que todavía
> no me ha hecho falta. Uso analogías con software donde ayudan y aviso donde
> **no** son de fiar.

## El problema de fondo

En software normal tú escribes la función. En robótica de manipulación nadie
sabe escribir la función. "Coge el vaso" no se programa con `if`s: depende de
dónde esté el vaso, de la luz, de si está resbaladizo, de mil cosas. Hay dos
salidas:

1. **Reinforcement Learning (RL)** — dejas al robot probar millones de veces y
   le das una recompensa cuando acierta. Aprende solo. Problema: millones de
   intentos en un robot real significa romper el robot, y en simulación
   significa que lo aprendido casi nunca transfiere bien al mundo real.
2. **Imitation Learning (IL)** — tú haces la tarea unas cuantas veces
   controlando el robot a mano, grabas todo, y entrenas un modelo a copiarte.
   Muchísimo más barato. Es lo que hace LeRobot y lo que hacemos aquí.

IL es, en el fondo, **aprendizaje supervisado**: tienes pares (entrada, salida
correcta) y ajustas un modelo. La entrada es "lo que el robot ve y siente", la
salida es "lo que yo hice". Nada exótico. Lo difícil viene de la estructura
temporal, que es donde está el problema real.

## Vocabulario mínimo

### Observación (`observation`)

Todo lo que el robot percibe en un instante. Típicamente dos familias:

- **Imágenes** — una o varias cámaras. En LeRobot son claves tipo
  `observation.images.top`, `observation.images.wrist`.
- **Estado propioceptivo** (`observation.state`) — dónde están sus propias
  articulaciones. Un vector de números.

**Propiocepción** es la palabra de robótica para "el sentido de dónde está tu
propio cuerpo". Cierra los ojos y tócate la nariz: eso es propiocepción. Para un
robot es literalmente leer el encoder de cada motor.

### Acción (`action`)

Lo que el robot ordena a sus motores en ese instante.

**Aquí está la primera trampa conceptual, y es importante:** para casi todos los
robots de LeRobot la acción **no** es una velocidad ni un par de fuerza. Es la
*posición objetivo* de cada articulación. O sea: "quiero que el codo esté a 45°".
Un controlador de bajo nivel dentro del servo se encarga de llegar ahí.

Consecuencia práctica: `action[t]` y `observation.state[t]` viven en el mismo
espacio y tienen casi los mismos números. `state` es "dónde estoy realmente",
`action` es "dónde te he pedido que estés". La diferencia entre ambos es el
error de seguimiento del motor. Cuando veas un dataset donde `action` y `state`
parecen casi idénticos desplazados un paso, no está roto — es exactamente lo
esperado.

### Grados de libertad (DoF)

Cuántos números hacen falta para describir la pose. Un brazo SO-100 tiene 6 (5
articulaciones + pinza). Nuestra plataforma pan/tilt tiene **2**: girar y
levantar. Por eso es un buen primer proyecto — el espacio de acciones es un
vector de 2 números y lo puedes graficar entero en una hoja.

### Episodio (`episode`)

Un intento completo de la tarea, de principio a fin. "Cojo el cubo y lo meto en
la caja" = un episodio. Si grabas 50 veces, tienes 50 episodios.

Analogía software: un episodio se parece a un **test de integración**, no a un
unit test. Tiene estado que evoluciona y solo tiene sentido completo.

### Frame / paso de tiempo (`timestep`)

Un episodio es una secuencia de frames. Cada frame es una foto del mundo:
`(observation, action, timestamp)`. Si grabas a 30 Hz durante 10 segundos,
tienes 300 frames.

### Demostración

Un episodio grabado **por un humano** teleoperando. Es el dato de
entrenamiento. "50 demostraciones" = 50 episodios que hiciste tú a mano.

### Teleoperación

Controlar el robot en tiempo real mientras se graba. Puede ser con teclado (lo
que haremos), con un mando, o con un "robot líder" — un brazo gemelo que mueves
con la mano y el robot real copia. Esto último es el estándar en manipulación
porque produce movimientos mucho más naturales y continuos que un teclado.

### Política (`policy`)

El modelo entrenado. Es una función:

```
política(observación) -> acción
```

Ese es todo el concepto. Un `nn.Module` de PyTorch cuya entrada son imágenes +
estado y cuya salida son números para los motores. Cuando leas "entrenar una
política", traduce a "entrenar una red que mapea lo que ve a lo que hace".

### Rollout

Ejecutar la política y ver qué pasa. En bucle: observar → predecir acción →
ejecutar → observar de nuevo. Es la fase de inferencia. En simulación se puede
hacer mil veces; en real cada rollout es tiempo de reloj y riesgo de romper algo.

## Por qué IL es más difícil de lo que parece

Si es aprendizaje supervisado, ¿por qué no funciona a la primera? Tres razones
concretas:

### 1. Compounding error (error acumulativo)

Es **el** problema del campo. Tu modelo se entrena con estados que aparecieron
en *tus* demostraciones. En ejecución comete un error pequeño y acaba en un
estado ligeramente distinto a todo lo que vio. Ahí predice peor, se desvía más,
y en pocos pasos está en territorio completamente desconocido haciendo
disparates.

Analogía: un modelo entrenado solo con código bien formateado que, en cuanto ve
una indentación rara, empieza a generar basura — y su propia basura es su
siguiente entrada.

Esto tiene nombre formal: *distribution shift* entre la distribución de estados
del experto y la de la política. Es la razón de ser de algoritmos como DAgger, y
parte de por qué ACT y Diffusion Policy predicen secuencias enteras en vez de
acciones sueltas.

### 2. Multimodalidad

Para rodear un obstáculo, izquierda y derecha son ambas correctas. Si entrenas
con error cuadrático medio, el modelo aprende **el promedio** de las dos
demostraciones: ir recto, y chocar. El promedio de dos soluciones buenas es una
solución pésima.

Esta es la razón técnica de que exista Diffusion Policy: un modelo de difusión
representa una distribución multimodal en vez de un solo punto, así que puede
decir "izquierda o derecha" en vez de "el promedio de las dos".

### 3. Demostraciones inconsistentes

Si en unas demos vas rápido y en otras despacio, o cambias de estrategia a
mitad, el modelo recibe señales contradictorias para observaciones casi
idénticas. En IL la **consistencia** de las demos importa más que la cantidad.
Esto es directamente relevante para la Parte B: al grabar las 50-100 demos con
el pan/tilt, tengo que hacer la tarea siempre igual.

## Las dos políticas que vamos a usar

### ACT — Action Chunking with Transformers

La idea central es **action chunking**: en vez de predecir la siguiente acción,
predice las próximas ~100 de golpe y las ejecuta en bloque.

Por qué ayuda: si predices de una en una, tienes 100 oportunidades de desviarte
en 100 pasos (problema 1). Si predices el bloque entero de una vez, el error no
se realimenta dentro del bloque. Además captura la intención a largo plazo en
vez de reaccionar frame a frame.

Arquitectura: un CVAE (Conditional Variational Autoencoder) con un transformer.
El backbone visual es un ResNet18 por cámara; el transformer fusiona las
imágenes con el estado y emite el chunk de acciones.

En ejecución se usa *temporal ensembling*: en cada instante hay varios chunks
solapados que opinan sobre la acción actual, y se promedian con pesos
exponenciales. Suaviza mucho el movimiento.

### Diffusion Policy

Genera la secuencia de acciones con el mismo mecanismo con que Stable Diffusion
genera imágenes: parte de ruido puro y lo va limpiando en N pasos, condicionado
por la observación.

Su ventaja es la del punto 2: modela distribuciones multimodales de forma
nativa. Su desventaja es el coste — hay que correr el bucle de denoising en cada
inferencia, así que es más lenta que ACT.

**Regla práctica:** empieza por ACT. Es más rápida de entrenar, más rápida en
inferencia, y tiene menos hiperparámetros que ajustar mal.

## Simulación vs. real

**Sim** es un mundo de física simulada (MuJoCo, PyBullet). Ventajas: gratis,
rápido, reseteable, y puedes medir la tasa de éxito automáticamente sobre
cientos de episodios sin estar delante.

**Sim-to-real gap**: lo que funciona en simulación normalmente no funciona en el
robot real. La física no es idéntica, las texturas no son idénticas, y el modelo
se agarra a detalles del simulador que no existen en el mundo.

Para la Parte A esto no importa: el objetivo es entender el pipeline, y sim es
el sitio correcto para eso. Para la Parte B trabajamos directo en real, sin sim,
y por eso la tarea tiene que ser muy simple.

## Nota sobre "no espero que funcione bien"

Conviene calibrar expectativas desde ya. Con 50 demostraciones de una tarea
simple, un modelo que acierte a veces ya es un resultado válido. Los papers de
ACT usan del orden de 50 demos por tarea **pero** con teleoperación bimanual de
alta calidad y hardware caro. Con un pan/tilt de 2 servos y un teclado, lo que
buscamos es que el pipeline entero funcione de punta a punta y que la curva de
pérdida baje de forma limpia. Eso es el entregable, no una tasa de éxito alta.
