/*
 * esp32_pantilt — plataforma pan/tilt de 2 servos controlada por serial
 *
 * Parte B del proyecto de imitation learning. El host (Python) manda posiciones
 * objetivo y lee la posicion real; esa distincion es justamente la que hace
 * falta para grabar datasets al estilo LeRobot:
 *
 *   action            = posicion OBJETIVO que mando el host   (target)
 *   observation.state = posicion REAL donde esta el servo      (current)
 *
 * Por eso el firmware no salta al objetivo de golpe: interpola hacia el con una
 * velocidad maxima (slew rate). Asi `state` persigue a `action` con un retardo
 * fisico realista, igual que en un brazo de verdad. Si saltara de golpe, state y
 * action serian identicos y el dataset no tendria nada que aprender.
 *
 * PROTOCOLO (ASCII por linea, 115200 baudios, terminador '\n')
 *
 *   Host -> ESP32
 *     M <pan> <tilt>   Mover: fija la posicion objetivo en grados (float)
 *     S                Estado: pide una lectura
 *     H                Home: vuelve al centro (90, 90)
 *     V <deg_por_s>    Velocidad maxima de slew
 *     E <0|1>          Enable: 0 suelta los servos (deja de dar pulsos)
 *     P                Ping -> responde "PONG"
 *
 *   ESP32 -> Host
 *     READY <pan> <tilt> <fw_version>       una vez al arrancar
 *     S <pan> <tilt> <tgt_pan> <tgt_tilt> <millis>    respuesta a S y a M
 *     PONG
 *     ERR <mensaje>
 *
 * Toda respuesta a un comando es exactamente UNA linea, lo cual permite al host
 * hacer request/response sincrono sin ambiguedad. No hay streaming automatico a
 * proposito: el host marca el ritmo de muestreo y asi el timestamp del dataset
 * es el del host, que es el mismo reloj que timestampea la webcam.
 *
 * CABLEADO
 *   Servo pan  -> GPIO 13
 *   Servo tilt -> GPIO 14
 *
 *   Evito GPIO 12 aunque parezca libre: es un pin de strapping (MTDI) y si el
 *   servo lo deja alto en el arranque, el ESP32 fija el voltaje de la flash a
 *   1.8V y no bootea. GPIO 13 y 14 no tienen ese problema.
 *
 * ALIMENTACION — importante
 *   Los servos NO se alimentan del pin 5V/3V3 del ESP32. Un SG90 pide picos de
 *   ~700 mA al arrancar; el regulador de la placa no da eso y provocas
 *   brownouts (la placa se reinicia sola en cuanto mueves los dos a la vez).
 *
 *   Fuente externa de 5V -> V+ de los dos servos
 *   GND de la fuente     -> GND del ESP32     <-- masa comun, imprescindible
 *   Senal de cada servo  -> su GPIO
 *
 *   Sin la masa comun no hay referencia de tension y los servos hacen cosas
 *   aleatorias. Es el fallo numero uno con servos.
 *
 * DEPENDENCIA
 *   Libreria "ESP32Servo" de Kevin Harrington (Gestor de Librerias del IDE).
 *   La libreria Servo.h estandar de Arduino NO funciona en ESP32.
 */

#include <ESP32Servo.h>

// ---------------------------------------------------------------- config ----

static const char *FW_VERSION = "1.0.0";

static const int PIN_PAN  = 13;
static const int PIN_TILT = 14;

// Limites mecanicos. Ajustar a tu montaje ANTES de mover nada: si el soporte
// choca, el servo sigue empujando y se quema el engranaje. Empieza estrecho.
static const float PAN_MIN  = 20.0f;
static const float PAN_MAX  = 160.0f;
static const float TILT_MIN = 45.0f;
static const float TILT_MAX = 135.0f;

static const float HOME_PAN  = 90.0f;
static const float HOME_TILT = 90.0f;

// Ancho de pulso en microsegundos. 500-2400 es el rango tipico de un SG90.
// Si tu servo no llega a los extremos, o hace ruido al llegar, ajusta esto.
static const int PULSE_MIN_US = 500;
static const int PULSE_MAX_US = 2400;

// Velocidad maxima por defecto, en grados por segundo. 180 significa que
// recorre el rango entero en un segundo. Bajala si el movimiento es brusco.
static const float DEFAULT_SLEW_DPS = 180.0f;

// Periodo del bucle de control. 5 ms = 200 Hz, muy por encima de los 50 Hz a
// los que el servo acepta pulsos, asi que la interpolacion sale suave.
static const unsigned long CONTROL_PERIOD_MS = 5;

// ----------------------------------------------------------------- estado ---

Servo servoPan;
Servo servoTilt;

static float curPan  = HOME_PAN;   // posicion real (la que reportamos como state)
static float curTilt = HOME_TILT;
static float tgtPan  = HOME_PAN;   // posicion objetivo (la que mando el host = action)
static float tgtTilt = HOME_TILT;

static float slewDps = DEFAULT_SLEW_DPS;
static bool  enabled = true;

static unsigned long lastControlMs = 0;

static char   lineBuf[64];
static size_t lineLen = 0;

// ---------------------------------------------------------------- helpers ---

static float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

/* Mueve `cur` hacia `tgt` sin pasarse de `maxStep`. Es todo el slew rate
   limiter: la diferencia entre esto y `cur = tgt` es lo que da al dataset una
   dinamica que aprender. */
static float slewToward(float cur, float tgt, float maxStep) {
  float d = tgt - cur;
  if (d >  maxStep) d =  maxStep;
  if (d < -maxStep) d = -maxStep;
  return cur + d;
}

static void writeServos() {
  if (!enabled) return;
  servoPan.write((int)lroundf(curPan));
  servoTilt.write((int)lroundf(curTilt));
}

static void sendState() {
  // 2 decimales: suficiente resolucion y mantiene la linea corta, que a 115200
  // baudios y 30 Hz importa (ver docs/03 sobre el presupuesto de tiempo).
  Serial.print("S ");
  Serial.print(curPan, 2);   Serial.print(' ');
  Serial.print(curTilt, 2);  Serial.print(' ');
  Serial.print(tgtPan, 2);   Serial.print(' ');
  Serial.print(tgtTilt, 2);  Serial.print(' ');
  Serial.println(millis());
}

// --------------------------------------------------------------- comandos ---

static void handleLine(char *s) {
  while (*s == ' ') s++;
  if (*s == '\0') return;

  const char cmd = *s++;

  switch (cmd) {
    case 'M': {
      float p, t;
      if (sscanf(s, "%f %f", &p, &t) != 2) {
        Serial.println("ERR M necesita dos numeros");
        return;
      }
      // El clamp va aqui, en el firmware, no en el host: el firmware es la
      // ultima linea de defensa del hardware y no debe fiarse del que manda.
      tgtPan  = clampf(p, PAN_MIN,  PAN_MAX);
      tgtTilt = clampf(t, TILT_MIN, TILT_MAX);
      sendState();
      return;
    }

    case 'S':
      sendState();
      return;

    case 'H':
      tgtPan  = HOME_PAN;
      tgtTilt = HOME_TILT;
      sendState();
      return;

    case 'V': {
      float v;
      if (sscanf(s, "%f", &v) != 1 || v <= 0.0f) {
        Serial.println("ERR V necesita un numero positivo");
        return;
      }
      slewDps = v;
      sendState();
      return;
    }

    case 'E': {
      int e;
      if (sscanf(s, "%d", &e) != 1) {
        Serial.println("ERR E necesita 0 o 1");
        return;
      }
      enabled = (e != 0);
      if (enabled) {
        servoPan.attach(PIN_PAN,   PULSE_MIN_US, PULSE_MAX_US);
        servoTilt.attach(PIN_TILT, PULSE_MIN_US, PULSE_MAX_US);
      } else {
        // Sin pulsos el servo deja de mantener la posicion. Util para que no
        // zumben ni se calienten entre demostraciones.
        servoPan.detach();
        servoTilt.detach();
      }
      sendState();
      return;
    }

    case 'P':
      Serial.println("PONG");
      return;

    default:
      Serial.print("ERR comando desconocido: ");
      Serial.println(cmd);
      return;
  }
}

static void pollSerial() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      // Linea demasiado larga: la descarto entera en vez de procesar un
      // prefijo truncado, que seria peor (un "M 90 9" moveria mal el tilt).
      lineLen = 0;
      Serial.println("ERR linea demasiado larga");
    }
  }
}

// ------------------------------------------------------------------ setup ---

void setup() {
  Serial.begin(115200);

  // El ESP32 tiene 4 timers hardware; ESP32Servo los reparte entre los servos.
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);

  servoPan.setPeriodHertz(50);   // 50 Hz es el estandar de servo analogico
  servoTilt.setPeriodHertz(50);
  servoPan.attach(PIN_PAN,   PULSE_MIN_US, PULSE_MAX_US);
  servoTilt.attach(PIN_TILT, PULSE_MIN_US, PULSE_MAX_US);

  writeServos();

  Serial.print("READY ");
  Serial.print(curPan, 2);  Serial.print(' ');
  Serial.print(curTilt, 2); Serial.print(' ');
  Serial.println(FW_VERSION);
}

void loop() {
  pollSerial();

  const unsigned long now = millis();
  const unsigned long dt = now - lastControlMs;   // resta unsigned: sobrevive
                                                  // al desbordamiento de millis()
                                                  // a los ~49 dias
  if (dt >= CONTROL_PERIOD_MS) {
    lastControlMs = now;
    const float maxStep = slewDps * (dt / 1000.0f);
    curPan  = slewToward(curPan,  tgtPan,  maxStep);
    curTilt = slewToward(curTilt, tgtTilt, maxStep);
    writeServos();
  }
}
