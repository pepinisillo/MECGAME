# MECGAME — memes por expresión

Al hacerse una pose frente a la webcam, se muestra el meme que le corresponde. Está pensado para ser
usado en vivo durante una partida que está siendo grabada.

## Cómo son reconocidas las poses

En cada cuadro son combinadas cuatro señales, y el meme es decidido por una red neuronal pequeña
(entrenada **con las tomas del propio usuario**):

| Señal | Qué es analizado | Ejemplos |
|---|---|---|
| Cara (MediaPipe) | 52 gestos faciales + giro de la cabeza | grito, sonrisa, pujar |
| Manos (MediaPipe) | si cada dedo está estirado o doblado, hacia dónde apunta la mano, dónde se ubica respecto a la cara | pulgar arriba, dedo grosero, manos en las orejas |
| Brazos (MediaPipe) | hombros, codos y muñecas | mano detrás de la cabeza |
| Visión (DINOv2 en la GPU) | recortes del cuerpo, la cara y la zona de cada mano | **cartera, celular**, lengua de fuera, cualquier objeto |

La señal de visión fue añadida para los objetos: con los landmarks solo es descrita la geometría del
cuerpo, por lo que una mano sosteniendo una cartera y una mano vacía eran vistas como iguales.

Lo que no se parece a nada de lo grabado es tratado como `neutral` (sin meme).

## Instalación

Se requiere Python 3.9 o más nuevo.

```bash
pip install -r requirements.txt
```

- **Linux + NVIDIA**: con ese comando ya es instalado PyTorch con CUDA.
- **Windows + NVIDIA**: PyTorch con CUDA debe ser instalado antes, siguiendo https://pytorch.org/get-started/locally/
  (con el `pip install torch` normal en Windows solo se instala la versión para CPU).
- `opencv-python` no debe ser instalado aparte: MediaPipe ya incluye su propio OpenCV y dos copias entran en conflicto.

La primera vez, los modelos son descargados automáticamente (MediaPipe ~20 MB, DINOv2 ~90 MB).

## Uso

1. Las imágenes son colocadas en `memes/`. El nombre del archivo es tomado como el nombre de la expresión
   (`cartera.png`, `dedo_grosero.jpg`, ...). Opcional: `neutral.png` es mostrado en el modo "solo meme"
   cuando no se detecta ninguna pose.
2. Se graban las poses:
   ```bash
   python memes_webcam.py calibrate --cam 10
   ```
   Para cada expresión: con `ESPACIO` se graba una toma (6 s), con `X` es borrada la última y con `ENTER`
   se pasa a la siguiente. Al terminar, las tomas son procesadas y el modelo es entrenado automáticamente.
3. Se revisa qué poses son confundidas:
   ```bash
   python memes_webcam.py test
   ```
   El modelo es entrenado varias veces dejando tomas fuera y es evaluado con ellas, de modo que se mide
   lo que ocurre con poses no vistas durante el entrenamiento. `acierto` indica cuántos cuadros son
   clasificados correctamente; `se activaría`, cuántos además superan los umbrales para mostrar el meme;
   en neutral, el dato relevante es `activaciones falsas`.
4. Se ejecuta el juego:
   ```bash
   python memes_webcam.py run --cam 10                # webcam con el meme encima
   python memes_webcam.py run --cam 10 --solo --mini  # solo el meme, ventana chica siempre encima
   ```

`calibrate` puede ser ejecutado de nuevo en cualquier momento para agregar tomas (`--only cartera celular`);
las tomas son acumuladas y el modelo es reentrenado automáticamente la siguiente vez que se ejecuta `run`.

### Teclas en `run`

| Tecla | Acción |
|---|---|
| `1` / `2` | meme encima de la webcam / solo el meme (cuadrado) |
| `p` | mini-webcam en el modo 2 |
| `d` | debug: barras de probabilidad, lo detectado, tiempo de análisis |
| `n` | "eso NO era un meme": los últimos segundos son guardados como neutral y el modelo es reentrenado al momento |
| `t` | modo mini (ventana chica, siempre encima) |
| `f` | pantalla completa |
| `q` / `ESC` | salir |

## Recomendaciones para un buen reconocimiento

- **Neutral es la clase más importante.** En ella es incluido todo lo que *no* es meme: jugar, hablar,
  reír, tomar agua, rascarse. Mientras más variado sea, menos activaciones falsas son producidas. Se
  recomienda grabarlo durante una partida real:
  ```bash
  python memes_webcam.py calibrate --cam 10 --only neutral --seconds 120
  ```
  (se presiona `ESPACIO`, se cambia al juego y se juega con normalidad durante dos minutos, sin hacer poses).
- **Objetos (cartera, celular):** en neutral deben ser incluidas las manos vacías en la misma posición en
  que es sostenido el objeto. Así es aprendido que lo importante es el objeto y no la postura.
- **2–3 tomas por pose**, variando un poco la distancia, el ángulo y la mano. Con `MIRROR = True` una pose
  es considerada válida tanto con la mano izquierda como con la derecha.
- **Las tomas deben ser grabadas con la misma luz y el mismo encuadre que serán usados al jugar.**
- Si un meme es activado sin motivo durante la partida, se presiona `n` en ese momento (la ventana de
  memes debe tener el foco). Si un meme no es activado, se graba una toma más de esa pose.
- Un gesto nunca grabado pero parecido a un meme (un puño frente a "pulgar arriba") puede activarlo:
  esto es corregido con `n` o grabándolo dentro de neutral.
- En `calibrate` y en debug es dibujado lo detectado (esqueleto de manos, brazos y los recortes analizados
  por el modelo de visión). Si en una pose la mano no es detectada, no hay problema: la zona sigue siendo
  analizada a partir de la muñeca.

## Ajustes

Se encuentran al inicio de `memes_webcam.py`. Los más útiles:

| Ajuste | Efecto |
|---|---|
| `ON_MIN` | seguridad mínima para activar un meme. Debe ser aumentado si los memes se activan solos. |
| `HOLD` | tiempo que la pose debe ser sostenida antes de que aparezca el meme. |
| `MIN_SHOW` | tiempo mínimo que un meme es mantenido en pantalla. |
| `OVERLAY_DELAY` | retraso extra antes de cubrir la webcam, para que la pose alcance a ser vista. |
| `FADE_IN`, `FADE_OUT` | velocidad de los fundidos. |
| `REJECT_LO`, `REJECT_HI` | qué tan parecida a las tomas grabadas debe ser una pose (2 = hasta el doble de lo que las tomas se parecen entre sí; el valor es mostrado en debug como "lejanía"). Deben ser reducidos si hay activaciones de más, y aumentados si no hay activaciones. |
| `MAX_FPS` (o `--fps`) | análisis por segundo. Debe ser reducido si afecta el rendimiento del juego. |
| `VISION_MODEL` | modelo de visión (puede ser usado cualquiera de `timm`). |

Opciones de línea de comandos: `--no-vision` (solo cara/manos/brazos, mucho más ligero, pero los objetos
no son distinguidos), `--device cpu|cuda`, `--bg verde` (fondo para chroma key en el modo solo meme).
Con `python memes_webcam.py train` es forzado un reentrenamiento (por ejemplo, después de cambiar un ajuste del modelo).

## Archivos

```
memes_webcam.py
memes/               imágenes de los memes
dataset/<pose>/      tomas grabadas (.avi) y sus features ya calculadas (.npz)
modelo.pt            el modelo entrenado (es regenerado automáticamente)
models/              modelos de MediaPipe (son descargados automáticamente)
```

Para eliminar una toma mala basta con borrar su `.avi`. Las tomas guardadas con la tecla `n` son nombradas
con el prefijo `fp-`. También pueden ser colocados videos propios en `dataset/<pose>/`.

## Problemas comunes

- **La cámara no es abierta:** debe ser verificado que scrcpy siga en ejecución, que la cámara no esté siendo usada por OBS, y la salida de `ls -l /dev/video*`.
- **El programa va lento / el juego pierde FPS:** puede ser usado `--fps 12` o `--no-vision`.
- **"no hay GPU disponible para torch":** PyTorch fue instalado sin CUDA; debe ser reinstalado desde pytorch.org.
- **Al venir de la versión anterior:** `dataset.npz` ya no es utilizado; las poses deben ser calibradas de nuevo.
