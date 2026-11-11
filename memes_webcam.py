#!/usr/bin/env python3
"""
MEMES POR EXPRESION v3
webcam + MediaPipe (cara, manos, brazos) + modelo de vision (objetos: cartera, celular...) + red neuronal

Instalacion (Python 3.9+):
    pip install -r requirements.txt          (mediapipe, numpy, torch, timm)

Estructura:
    memes_webcam.py
    memes/      <- imagenes de los memes (nombre del archivo = nombre de la expresion). Opcional: neutral.png
    dataset/    <- tomas grabadas, una carpeta por expresion (es creada automaticamente)

Uso:
    python memes_webcam.py calibrate --cam 10              # se graban las poses (neutral primero)
    python memes_webcam.py test                            # se muestra cuales poses son confundidas entre si
    python memes_webcam.py run --cam 10                    # se juega
    python memes_webcam.py run --cam 10 --solo --mini      # solo memes (ventana cuadrada y pequena)
    python memes_webcam.py calibrate --only dedo_grosero   # se graban mas tomas de una sola expresion
    python memes_webcam.py calibrate --only neutral --seconds 120   # neutral largo: es grabado mientras se juega con normalidad
    python memes_webcam.py train                           # es forzado el reentrenamiento (normalmente es hecho automaticamente)

Teclas en "run":  1 = meme encima de webcam | 2 = solo meme (cuadrado) | p = mini-webcam en modo 2
                  d = debug | n = "eso NO era un meme" (es aprendido como neutral) | t = modo mini
                  f = pantalla completa | q/ESC = salir
"""
import argparse
import hashlib
import math
import sys
import threading
import time
import urllib.request
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"Falta una libreria ({e.name}). Todo puede ser instalado con:  pip install -r requirements.txt")

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# AJUSTES
# --------------------------------------------------------------------------
LONG_SIDE = 960                # lado largo de la imagen de trabajo (una camara 16:9 queda en 960x540)
SQ = 540                       # ventana modo 2 (solo meme): cuadrada, para imagenes cuadradas
MAX_FPS = 20                   # analisis por segundo como maximo (debe ser reducido si afecta el rendimiento del juego)

# -- que tan natural se ve el cambio entre memes
EMA_TAU = 0.15                 # seg. suavizado de probabilidades
ON_MIN = 0.60                  # probabilidad minima para activar un meme (debe ser aumentado si los memes se activan solos)
OFF_MIN = 0.35                 # por debajo de esto el meme activo es soltado (siempre menor que ON_MIN)
HOLD = 0.20                    # seg. que una expresion nueva debe mantenerse antes de activarse
RELEASE = 0.30                 # seg. por debajo de OFF_MIN antes de quitar el meme
MARGIN = 0.15                  # ventaja minima sobre el meme actual para cambiar a otro
MIN_SHOW = 0.90                # seg. minimos que un meme se queda en pantalla
REARM = 0.30                   # seg. minimos sin meme antes de activar otro
OVERLAY_DELAY = 0.15           # (modo 1) retraso extra antes de tapar la webcam, para que la pose alcance a ser vista
FADE_IN, FADE_OUT = 6.0, 3.5   # opacidad/seg

# -- poses que no se parecen a nada de lo grabado -> neutral
REJECT_LO, REJECT_HI = 2.0, 4.0   # lejania a las muestras grabadas: deben ser aumentados si nada es activado, reducidos si hay activaciones de mas
KNN_K = 5

# -- grabacion
REC_COUNTDOWN = 3
REC_SECONDS, REC_SECONDS_NEUTRAL = 6, 20
STORE_FPS = 12                 # cuadros por segundo que son guardados de cada toma
FP_SECONDS = 3.5               # tecla n: cuantos segundos hacia atras son guardados como neutral

# -- modelo
VISION_MODEL = "vit_small_patch14_dinov2.lvd142m"   # puede ser usado cualquier modelo de timm
VISION_SIZE = 224
MIRROR = True                  # la pose es considerada valida tanto con la mano izquierda como con la derecha
AUG_VARIANTS = 4               # versiones de cada cuadro al entrenar (espejo, cambios de luz y encuadre)
TRAIN_STEPS = 1200
NEUTRAL_WEIGHT = 2.5           # peso dado a neutral al entrenar (mas = menos activaciones falsas)
FEAT_VERSION = 3

BASE = Path(__file__).resolve().parent
MEMES_DIR = BASE / "memes"
DATASET = BASE / "dataset"
MODELS = BASE / "models"
MODEL_FILE = BASE / "modelo.pt"
MP_URL = "https://storage.googleapis.com/mediapipe-models/"
MODEL_URLS = {
    "face_landmarker.task": MP_URL + "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": MP_URL + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_full.task": MP_URL + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
BG_COLORS = {"negro": (0, 0, 0), "verde": (0, 255, 0), "magenta": (255, 0, 255)}

# --------------------------------------------------------------------------
# Formato de las features de landmarks (un vector por cuadro)
# --------------------------------------------------------------------------
BS_NAMES = (
    "_neutral browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft "
    "cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight "
    "eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight eyeWideLeft "
    "eyeWideRight jawForward jawLeft jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft "
    "mouthFrownRight mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight "
    "mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthSmileLeft "
    "mouthSmileRight mouthStretchLeft mouthStretchRight mouthUpperUpLeft mouthUpperUpRight noseSneerLeft "
    "noseSneerRight").split()
N_BS = len(BS_NAMES)


def _swap_lr(name):
    if name.endswith("Left"):
        return name[:-4] + "Right"
    if name.endswith("Right"):
        return name[:-5] + "Left"
    return name


BS_MIRROR = np.array([BS_NAMES.index(_swap_lr(n)) for n in BS_NAMES])

FACE_DIM = 1 + N_BS + 3     # presente | blendshapes | giro, cabeceo, inclinacion de la cabeza
POSE_DIM = 1 + 4            # presente | inclinacion de hombros, centro de hombros (x, y), ancho de hombros
ARM_DIM = 8                 # codo (x, y), muneca (x, y), visibilidad codo y muneca, muneca respecto a la cara (x, y)
HAND_DIM = 1 + 28           # presente | palma (x, y), indice (x, y), tamano, direccion (x, y), palma/dorso, dedos...
O_FACE = 0
O_POSE = O_FACE + FACE_DIM
O_ARMS = O_POSE + POSE_DIM
O_HANDS = O_ARMS + 2 * ARM_DIM
LM_DIM = O_HANDS + 2 * HAND_DIM
N_CROPS = 4                 # recortes para el modelo de vision: cuerpo, cara, zona de mano x2
FINGERS = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]              # (punta, articulacion de referencia)
FINGER_BONES = [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
TIP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
_MIRROR_NEG = ([O_FACE + 1 + N_BS, O_FACE + 3 + N_BS, O_POSE + 1, O_POSE + 2]
               + [O_ARMS + a * ARM_DIM + 6 for a in range(2)]
               + [O_HANDS + h * HAND_DIM + i for h in range(2) for i in (1, 3, 6)])


def mirror_lm(x):
    """Las mismas features como si la imagen estuviera volteada horizontalmente."""
    y = x.copy()
    y[..., 1:1 + N_BS] = x[..., 1:1 + N_BS][..., BS_MIRROR]
    y[..., _MIRROR_NEG] *= -1
    return y


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def need(module, pip_name=None):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"Falta la libreria '{pip_name or module}'. Puede ser instalada con:  pip install {pip_name or module}")


def ensure_models():
    MODELS.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        p = MODELS / name
        if not p.exists():
            print(f"Descargando {name} ...")
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(p)


def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def pick_device(arg):
    torch = need("torch")
    if arg in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


class Camera:
    """La camara es leida en un hilo aparte y siempre es entregado el cuadro mas reciente (ya en espejo)."""

    def __init__(self, idx):
        self.cap, self.is_file = self._open(idx)
        ok, first = self.cap.read()
        if not ok:
            sys.exit(f"La camara {idx} fue abierta, pero no es entregada ninguna imagen.")
        h, w = first.shape[:2]
        s = LONG_SIDE / max(w, h)
        self.W, self.H = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        self._latest, self.ended, self._run = (1, self._prep(first)), False, True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _open(idx):
        """idx: numero (0, 1, 10), ruta de dispositivo ('/dev/video10') o un archivo de video (para pruebas)."""
        if Path(str(idx)).is_file():
            cap = cv2.VideoCapture(str(idx))
            if not cap.isOpened():
                sys.exit(f"El video {idx} no pudo ser abierto.")
            return cap, True
        src = int(idx) if str(idx).isdigit() else idx
        if isinstance(src, int) and sys.platform.startswith("linux"):
            src = f"/dev/video{src}"
        cap = None
        for backend in ([cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform.startswith("linux") else [cv2.CAP_ANY]):
            cap = cv2.VideoCapture(src, backend)
            if cap.isOpened():
                break
        if cap is None or not cap.isOpened():
            sys.exit(f"La camara {src} no pudo ser abierta.\n"
                     "Debe ser verificado: que scrcpy siga en ejecucion, que la camara no este siendo usada por OBS, 'ls -l /dev/video*'.")
        if not str(src).endswith("video10"):          # una camara virtual (scrcpy) ya viene con su tamano
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, False

    def _prep(self, frame):
        if not self.is_file:
            frame = cv2.flip(frame, 1)                # espejo
        return cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def _loop(self):
        fails = 0
        while self._run:
            t = time.time()
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if self.is_file or fails > 100:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            self._latest = (self._latest[0] + 1, self._prep(frame))
            if self.is_file:
                time.sleep(max(0.0, 1 / self.fps - (time.time() - t)))
        self.ended = True

    def latest(self):
        """(numero de cuadro, imagen). La imagen no debe ser modificada: debe ser copiada antes de dibujar encima."""
        return self._latest

    def close(self):
        self._run = False
        self._thread.join(timeout=1)
        self.cap.release()


def load_memes(W, H):
    """{nombre: (imagen float32 HxWx3, mascara HxWx1, caja)} ajustado a un lienzo WxH ('contain').
    caja = (x0, y0, x1, y1) si la imagen no tiene transparencia (puede ser pegada directo), si no None."""
    if not MEMES_DIR.exists():
        MEMES_DIR.mkdir()
        sys.exit(f"La carpeta {MEMES_DIR} fue creada. Las imagenes deben ser colocadas ahi y el programa debe ser ejecutado de nuevo.")
    memes = {}
    for p in sorted(MEMES_DIR.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"(saltando {p.name}: no se pudo leer)")
            continue
        if im.dtype != np.uint8:
            im = (im.astype(np.float32) / np.iinfo(im.dtype).max * 255).astype(np.uint8)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 4:
            a, bgr = im[:, :, 3].astype(np.float32) / 255.0, im[:, :, :3]
        else:
            a, bgr = np.ones(im.shape[:2], np.float32), im
        s = min(W / bgr.shape[1], H / bgr.shape[0])
        nw, nh = max(1, int(bgr.shape[1] * s)), max(1, int(bgr.shape[0] * s))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        solid = float(a.min()) > 0.999          # sin transparencia: mascara exacta de unos
        a = np.ones((nh, nw, 1), np.float32) if solid else cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA).reshape(nh, nw, 1)
        canvas = np.zeros((H, W, 3), np.float32)
        mask = np.zeros((H, W, 1), np.float32)
        x0, y0 = (W - nw) // 2, (H - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = bgr
        mask[y0:y0 + nh, x0:x0 + nw] = a
        memes[p.stem] = (canvas, mask, (x0, y0, x0 + nw, y0 + nh) if solid else None)
    if not [n for n in memes if n != "neutral"]:
        sys.exit(f"No fueron encontradas imagenes en {MEMES_DIR}")
    return memes


# --------------------------------------------------------------------------
# Tomas grabadas: dataset/<expresion>/<fecha>.avi  (+ .npz con las features ya calculadas)
# --------------------------------------------------------------------------
def takes_of(name):
    d = DATASET / name
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT) if d.is_dir() else []


def dataset_names():
    if not DATASET.is_dir():
        return []
    found = sorted(d.name for d in DATASET.iterdir() if d.is_dir() and d.name != "neutral" and takes_of(d.name))
    return (["neutral"] if takes_of("neutral") else []) + found


def delete_take(path):
    for p in (path, path.with_suffix(".npz")):
        if p.exists():
            p.unlink()


class TakeWriter:
    def __init__(self, name, fps, size, prefix=""):
        d = DATASET / name
        d.mkdir(parents=True, exist_ok=True)
        stamp, k = time.strftime("%Y%m%d-%H%M%S"), 0
        self.path = d / f"{prefix}{stamp}.avi"
        while self.path.exists():
            k += 1
            self.path = d / f"{prefix}{stamp}-{k}.avi"
        self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), size)
        if not self.vw.isOpened():
            sys.exit(f"El archivo de video {self.path} no pudo ser creado")
        self.vw.set(cv2.VIDEOWRITER_PROP_QUALITY, 92)
        self.n = 0

    def write(self, frame):
        self.vw.write(frame)
        self.n += 1

    def close(self):
        self.vw.release()
        return self.path


def read_take(path):
    """Son generados pares (cuadro, dt) de un video guardado, a STORE_FPS como maximo y con lado largo LONG_SIDE."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    fps = fps if 0.5 <= fps <= 240 else float(STORE_FPS)
    step = max(1, int(round(fps / STORE_FPS)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            if max(w, h) != LONG_SIDE:
                s = LONG_SIDE / max(w, h)
                frame = cv2.resize(frame, (int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2),
                                   interpolation=cv2.INTER_AREA)
            yield frame, step / fps
        i += 1
    cap.release()


# --------------------------------------------------------------------------
# Features de landmarks: cara + cabeza + brazos + manos (con estado de cada dedo)
# --------------------------------------------------------------------------
class Obs:
    """Lo que fue visto en un cuadro: vector de features, recortes para vision y puntos para dibujar."""
    __slots__ = ("x", "boxes", "has_face", "has_pose", "n_hands", "hand_pts", "arm_pts")


def _unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-6)


class Extractor:
    def __init__(self):
        self.mp = need("mediapipe")
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.mpt, self.vision = mpt, vision
        ensure_models()
        self.pool = ThreadPoolExecutor(3)
        self.models = []
        self.reset()

    def reset(self):
        """Modelos nuevos, sin memoria del video anterior."""
        for m in self.models:
            m.close()
        V, base = self.vision, lambda f: self.mpt.BaseOptions(model_asset_path=str(MODELS / f))
        video = V.RunningMode.VIDEO
        self.models = [
            V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
                base_options=base("face_landmarker.task"), running_mode=video, num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True)),
            V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
                base_options=base("hand_landmarker.task"), running_mode=video, num_hands=2,
                min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4)),
            V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
                base_options=base("pose_landmarker_full.task"), running_mode=video)),
        ]
        self.t0, self.ts = time.time(), 0
        self.anchor, self.anchor_ts = None, -10 ** 9
        self.bs_order = None

    def close(self):
        for m in self.models:
            m.close()
        self.models = []
        self.pool.shutdown(wait=False)

    def __call__(self, frame, dt=None):
        """dt=None: tiempo real (camara). dt=segundos desde el cuadro anterior: video guardado."""
        ts = int((time.time() - self.t0) * 1000) if dt is None else self.ts + int(dt * 1000)
        self.ts = ts = max(ts, self.ts + 1)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        jobs = [self.pool.submit(m.detect_for_video, img, ts) for m in self.models]
        fr, hr, pr = [j.result() for j in jobs]
        H, W = frame.shape[:2]
        return self._features(fr, hr, pr, W, H)

    def _features(self, fr, hr, pr, W, H):
        o = Obs()
        x = np.zeros(LM_DIM, np.float32)
        size = np.array([W, H], np.float32)

        # --- cara: blendshapes + orientacion de la cabeza
        face = None
        o.has_face = bool(fr.face_landmarks and fr.face_blendshapes)
        if o.has_face:
            pts = np.array([(p.x, p.y) for p in fr.face_landmarks[0]], np.float32) * size
            c = (pts.min(0) + pts.max(0)) / 2
            s = max(float(np.linalg.norm(pts[10] - pts[152])), 2.0 * float(np.linalg.norm(pts[33] - pts[263])), 8.0)
            face = (float(c[0]), float(c[1]), s)
            cats = fr.face_blendshapes[0]
            if self.bs_order is None:
                names = [c_.category_name for c_ in cats]
                self.bs_order = [names.index(n) if n in names else len(names) for n in BS_NAMES]
            scores = np.array([c_.score for c_ in cats] + [0.0], np.float32)
            x[O_FACE] = 1
            x[O_FACE + 1:O_FACE + 1 + N_BS] = scores[self.bs_order]
            mats = getattr(fr, "facial_transformation_matrixes", None)
            if mats is not None and len(mats):
                R = np.array(mats[0], np.float32).reshape(4, 4)[:3, :3]
                x[O_FACE + 1 + N_BS:O_FACE + FACE_DIM] = (
                    math.atan2(R[0, 2], R[2, 2]),                               # giro (izq/der)
                    math.atan2(-R[1, 2], math.hypot(R[0, 2], R[2, 2])),         # cabeceo (arriba/abajo)
                    math.atan2(R[1, 0], R[1, 1]))                               # inclinacion

        # --- cuerpo: hombros, codos y munecas
        o.has_pose = bool(pr.pose_landmarks)
        P = vis = sw = None
        if o.has_pose:
            lm = pr.pose_landmarks[0]
            P = np.array([(p.x, p.y) for p in lm[:17]], np.float32) * size
            vis = np.array([p.visibility if p.visibility is not None else 1.0 for p in lm[:17]], np.float32)
            sw = max(float(np.linalg.norm(P[11] - P[12])), 8.0)

        # --- ancla: donde esta la cabeza y de que tamano (todo lo demas es medido respecto a ella)
        if face is not None:
            self.anchor, self.anchor_ts = face, self.ts
        elif self.anchor is None or self.ts - self.anchor_ts > 2000:    # cara perdida hace rato (tapada, de lado)
            if o.has_pose and vis[[0, 11, 12]].min() > 0.5 and sw > 0.1 * W:
                self.anchor = (float(P[0, 0]), float(P[0, 1]), 0.55 * sw)
            else:
                self.anchor = (W / 2, H * 0.42, 0.33 * min(W, H))
            self.anchor_ts = -10 ** 9                                   # provisional: es recalculado en cada cuadro
        ax, ay, s = self.anchor
        a_xy = np.array([ax, ay], np.float32)

        if o.has_pose:
            mid = (P[11] + P[12]) / 2
            lo, hi = (P[11], P[12]) if P[11, 0] <= P[12, 0] else (P[12], P[11])
            x[O_POSE:O_POSE + POSE_DIM] = (1, math.atan2(hi[1] - lo[1], hi[0] - lo[0]),
                                           *np.clip((mid - a_xy) / s, -4, 4), math.log(sw / s))
            for k, (i_s, i_e, i_w) in enumerate(((11, 13, 15), (12, 14, 16))):
                out = 1.0 if P[i_s, 0] >= mid[0] else -1.0             # hacia afuera del cuerpo = positivo
                ge, gw = (float(np.clip((vis[i] - 0.2) / 0.4, 0, 1)) for i in (i_e, i_w))
                e, w, wa = (P[i_e] - P[i_s]) / sw * ge, (P[i_w] - P[i_s]) / sw * gw, (P[i_w] - a_xy) / s * gw
                x[O_ARMS + k * ARM_DIM:O_ARMS + (k + 1) * ARM_DIM] = np.clip(
                    (e[0] * out, e[1], w[0] * out, w[1], vis[i_e], vis[i_w], wa[0], wa[1]), -4, 4)
            o.arm_pts = (P[11:17], vis[11:17])
        else:
            o.arm_pts = None

        # --- manos: posicion respecto a la cara + forma (independiente de rotacion y de que mano sea)
        regions, o.hand_pts = [], []
        for h, handed in zip(hr.hand_landmarks, hr.handedness):
            p = np.array([(q.x * W, q.y * H, q.z * W) for q in h], np.float32)
            hs = max(float(np.linalg.norm(p[9] - p[0])), 1.3 * float(np.linalg.norm(p[5] - p[17])), 4.0)
            palm = p[[0, 5, 9, 13, 17], :2].mean(0)
            sign = 1.0 if handed[0].category_name == "Right" else -1.0
            v1, v2 = p[5, :2] - p[0, :2], p[17, :2] - p[0, :2]
            nz = float(v1[0] * v2[1] - v1[1] * v2[0]) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6)
            ext = [(np.linalg.norm(p[t] - p[0]) - np.linalg.norm(p[r] - p[0])) / hs for t, r in FINGERS]
            straight = [float(_unit(p[b] - p[a]) @ _unit(p[t] - p[b])) for a, b, t in FINGER_BONES]
            tips = p[[4, 8, 12, 16, 20]]
            spread = [np.linalg.norm(tips[i] - tips[j]) / hs for i, j in TIP_PAIRS]
            feat = np.concatenate([[1.0], np.clip((palm - a_xy) / s, -4, 4), np.clip((p[8, :2] - a_xy) / s, -4, 4),
                                   [math.log(hs / s)], _unit(p[9, :2] - p[0, :2]), [nz * sign],
                                   np.clip(ext, -2, 2), straight, np.clip(spread, 0, 4)]).astype(np.float32)
            lo, hi = p[:, :2].min(0), p[:, :2].max(0)
            c = (lo + hi) / 2
            side = max(2.4 * hs, 1.5 * float((hi - lo).max()))
            regions.append((float(c[0]), float(c[1]), side, feat))
            o.hand_pts.append(p[:, :2])
        o.n_hands = len(regions)
        if o.has_pose:              # manos no detectadas por MediaPipe (p. ej. tapadas por un objeto): es usada la muneca
            for i_e, i_w in ((13, 15), (14, 16)):
                c = P[i_w] + _unit(P[i_w] - P[i_e]) * 0.4 * s
                inside = -0.1 * W < c[0] < 1.1 * W and -0.1 * H < c[1] < 1.1 * H
                if vis[i_w] < 0.6 or not inside or any(math.hypot(c[0] - r[0], c[1] - r[1]) < 0.8 * s for r in regions):
                    continue
                regions.append((float(c[0]), float(c[1]), 1.5 * s, None))
        regions = sorted(regions[:2], key=lambda r: r[0])
        for k, r in enumerate(regions):
            if r[3] is not None:
                x[O_HANDS + k * HAND_DIM:O_HANDS + (k + 1) * HAND_DIM] = r[3]

        # --- recortes para el modelo de vision: (centro x, centro y, lado) o None
        o.boxes = [(ax, ay + 0.8 * s, 4.4 * s),
                   (ax, ay + 0.12 * s, 1.7 * s) if o.has_face else None] + [r[:3] for r in regions]
        o.boxes += [None] * (N_CROPS - len(o.boxes))
        o.x = x
        return o


def draw_obs(img, o):
    """Es dibujado lo detectado (para el modo debug y la calibracion)."""
    colors = [(255, 200, 0), (0, 200, 255), (0, 255, 0), (0, 255, 0)]
    for b, col in zip(o.boxes, colors):
        if b is not None:
            cx, cy, side = b
            cv2.rectangle(img, (int(cx - side / 2), int(cy - side / 2)), (int(cx + side / 2), int(cy + side / 2)), col, 1)
    if o.arm_pts is not None:
        P, vis = o.arm_pts
        for a, b in ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5)):
            if vis[a] > 0.5 and vis[b] > 0.5:
                cv2.line(img, tuple(map(int, P[a])), tuple(map(int, P[b])), (255, 120, 255), 2, cv2.LINE_AA)
    for pts in o.hand_pts:
        for a, b in HAND_LINES:
            cv2.line(img, tuple(map(int, pts[a])), tuple(map(int, pts[b])), (60, 255, 60), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Vision: los recortes de la imagen (cuerpo, cara, zona de cada mano) son analizados por un modelo preentrenado.
# Asi es "visto" lo que no puede ser captado por los landmarks: una cartera, un celular, la lengua, etc.
# --------------------------------------------------------------------------
class Vision:
    def __init__(self, device):
        self.torch = need("torch")
        timm = need("timm")
        print(f"Cargando modelo de vision ({VISION_MODEL}) en {device} ...")
        try:
            net = timm.create_model(VISION_MODEL, pretrained=True, num_classes=0, img_size=VISION_SIZE)
            self.S = VISION_SIZE
        except TypeError:           # modelos de tamano fijo
            net = timm.create_model(VISION_MODEL, pretrained=True, num_classes=0)
            self.S = int(timm.data.resolve_model_data_config(net)["input_size"][-1])
        cfg = timm.data.resolve_model_data_config(net)
        self.net = net.eval().to(device)
        self.device = device
        self.mean = self.torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = self.torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.lock = threading.Lock()
        self.id = f"{VISION_MODEL}@{self.S}"
        self.dim = self.embed(np.zeros((1, self.S, self.S, 3), np.uint8)).shape[1]

    def embed(self, crops):
        """crops: (n, S, S, 3) uint8 BGR -> (n, dim) float32."""
        torch, out = self.torch, []
        with self.lock, torch.inference_mode():
            for i in range(0, len(crops), 64):
                t = torch.from_numpy(crops[i:i + 64]).to(self.device).permute(0, 3, 1, 2).flip(1).float() / 255
                t = (t - self.mean) / self.std
                with torch.autocast(self.device.split(":")[0], enabled=self.device.startswith("cuda")):
                    f = self.net.forward_features(t)
                f = f.float()
                if f.ndim == 3:     # transformer: token de clase + promedio de parches
                    k = getattr(self.net, "num_prefix_tokens", 1)
                    parts = [f[:, 0], f[:, k:].mean(1)] if k else [f.mean(1)]
                else:               # red convolucional
                    parts = [f.mean((2, 3))]
                out.append(torch.cat([torch.nn.functional.normalize(p, dim=1) for p in parts], 1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def crop(self, frame, half, box, flip=False, jit=None):
        cx, cy, side = box
        rot = 0.0
        if jit is not None:
            side *= jit[0]
            cx, cy, rot = cx + jit[1] * side, cy + jit[2] * side, jit[3]
        S = self.S
        sc = S / side
        co, si = sc * math.cos(rot), sc * math.sin(rot)
        M = np.array([[co, -si, S / 2 - (co * cx - si * cy)], [si, co, S / 2 - (si * cx + co * cy)]], np.float32)
        if flip:
            M[0] = (-M[0, 0], -M[0, 1], S - M[0, 2])
        src = frame
        if sc < 0.5:            # reduccion grande: se parte de la imagen a mitad de tamano (menos aliasing)
            src = half
            M[:, :2] *= 2
        return cv2.warpAffine(src, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def crops_of(self, frame, boxes, flip=False, rng=None):
        """Recortes de un cuadro. Con rng: encuadre e iluminacion cambiados al azar (aumento de datos)."""
        half = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        out = []
        for b in boxes:
            if b is None:
                continue
            jit = None if rng is None else (rng.uniform(0.87, 1.15), rng.uniform(-0.07, 0.07),
                                            rng.uniform(-0.07, 0.07), rng.uniform(-0.14, 0.14))
            out.append(self.crop(frame, half, b, flip, jit))
        if rng is not None and out:
            x = np.stack(out).astype(np.float32) / 255
            x = x ** rng.uniform(0.75, 1.35)                                   # gamma
            x = (x - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.08, 0.08)   # contraste y brillo
            g = x.mean(-1, keepdims=True)
            x = (g + (x - g) * rng.uniform(0.6, 1.3)) * rng.uniform(0.92, 1.08, 3)     # saturacion y tinte
            out = list(np.clip(x * 255, 0, 255).astype(np.uint8))
            if rng.random() < 0.3:
                out = [cv2.GaussianBlur(c, (0, 0), rng.uniform(0.5, 1.2)) for c in out]
        return out

    def embed_boxes(self, frame, boxes):
        """Embeddings de un cuadro en vivo -> (N_CROPS, dim), (N_CROPS,) presentes."""
        ok = np.array([b is not None for b in boxes], np.float32)
        emb = np.zeros((N_CROPS, self.dim), np.float32)
        emb[ok > 0] = self.embed(np.stack(self.crops_of(frame, boxes)))
        return emb, ok


def variant_plan(n):
    """[(espejo, aumento)] para cada version de un cuadro. La primera siempre es el original."""
    plan = [(False, False)] + ([(True, False)] if MIRROR else [])
    while len(plan) < n:
        plan.append((MIRROR and len(plan) % 2 == 1, True))
    return plan


def process_take(path, ex, vision, n_var):
    """Son calculadas y guardadas las features de una toma (landmarks + embeddings de cada version)."""
    plan = variant_plan(n_var) if vision else variant_plan(0)
    rng = np.random.default_rng(int(hashlib.sha1(path.name.encode()).hexdigest()[:8], 16))
    ex.reset()
    xs, oks, pending, embs = [], [], [], []

    def flush():
        if pending:
            embs.append(vision.embed(np.stack(pending)))
            pending.clear()

    for frame, dt in read_take(path):
        o = ex(frame, dt)
        xs.append(o.x)
        oks.append([b is not None for b in o.boxes])
        if vision:
            for flip, jitter in plan:
                pending += vision.crops_of(frame, o.boxes, flip, rng if jitter else None)
            if len(pending) >= 192:
                flush()
    if not xs:
        return None
    N, V = len(xs), len(plan)
    ok = np.array(oks, bool)
    emb = np.zeros((N, V, N_CROPS, vision.dim if vision else 1), np.float16)
    if vision:
        flush()
        emb[np.broadcast_to(ok[:, None, :], (N, V, N_CROPS))] = np.concatenate(embs)
    data = dict(x=np.array(xs, np.float32), emb=emb, ok=ok, flips=np.array([p[0] for p in plan]),
                jits=np.array([p[1] for p in plan]), key=np.array(cache_key(path, vision, n_var)))
    np.savez(path.with_suffix(".npz"), **data)
    return data


def cache_key(path, vision, n_var):
    st = path.stat()
    return f"v{FEAT_VERSION}|{st.st_size}|{st.st_mtime_ns}|{int(MIRROR)}|" + (f"{vision.id}|{n_var}" if vision else "-")


def load_take(path, vision, n_var):
    """Features guardadas de una toma si siguen vigentes; None si deben ser recalculadas."""
    cache = path.with_suffix(".npz")
    if not cache.exists():
        return None
    try:
        d = dict(np.load(cache))
    except Exception:
        return None
    key, want = str(d["key"]), cache_key(path, vision, n_var)
    same_video = key.split("|")[:4] == want.split("|")[:4]
    return d if key == want or (vision is None and same_video) else None


def load_dataset(vision, n_var, quiet=False):
    """Todas las tomas con sus features (son calculadas las que falten). -> nombres de clases, lista de tomas."""
    names = dataset_names()
    if "neutral" not in names:
        sys.exit("'neutral' no ha sido grabado. Debe ser ejecutado:  python memes_webcam.py calibrate --cam 10")
    if len(names) < 2:
        sys.exit("Solo ha sido grabado 'neutral'. Debe ser grabada al menos una expresion con calibrate.")
    old = BASE / "dataset.npz"
    if old.exists() and not quiet:
        print("Nota: dataset.npz es de una version anterior y ya no es usado (puede ser borrado).")
    takes, ex = [], None
    todo = [(n, p) for n in names for p in takes_of(n)]
    for i, (n, p) in enumerate(todo):
        d = load_take(p, vision, n_var)
        if d is None:
            if ex is None:
                ex = Extractor()
            if not quiet:
                print(f"  procesando toma {i + 1}/{len(todo)}: {n}/{p.name} ...", flush=True)
            d = process_take(p, ex, vision, n_var)
        if d is None or not len(d["x"]):
            print(f"  (toma vacia o ilegible, es ignorada: {p})")
            continue
        d["name"], d["path"] = n, p
        takes.append(d)
    if ex is not None:
        ex.close()
    names = [n for n in names if any(t["name"] == n for t in takes)]
    return names, takes


def assemble(takes, names, use_vision):
    """Las tomas son juntadas en arreglos de entrenamiento (una fila por cuadro y version)."""
    X, E, OK, y, tk, fr, base, flips = [], [], [], [], [], [], [], []
    for ti, t in enumerate(takes):
        n = len(t["x"])
        for v, flip in enumerate(t["flips"]):
            if not use_vision and t["jits"][v]:
                continue
            X.append(mirror_lm(t["x"]) if flip else t["x"])
            E.append(t["emb"][:, v] if use_vision else np.zeros((n, N_CROPS, 1), np.float16))
            OK.append(t["ok"] if use_vision else np.zeros_like(t["ok"]))
            y.append(np.full(n, names.index(t["name"])))
            tk.append(np.full(n, ti))
            fr.append(np.arange(n))
            base.append(np.full(n, not t["jits"][v]))
            flips.append(np.full(n, bool(flip)))
    cat = np.concatenate
    return dict(X=cat(X), E=cat(E), OK=cat(OK).astype(np.float32), y=cat(y), take=cat(tk), frame=cat(fr),
                base=cat(base), flip=cat(flips))


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


# --------------------------------------------------------------------------
# Clasificador: red neuronal pequena que fusiona todo + rechazo de poses desconocidas
# --------------------------------------------------------------------------
def build_net(n_cls, emb_dim, use_vision):
    torch = need("torch")
    nn, F = torch.nn, torch.nn.functional

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_vision = use_vision
            mlp = lambda i, h, o: nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))
            self.face = mlp(FACE_DIM - 1, 64, 48)
            self.pose = nn.Linear(POSE_DIM - 1, 16)
            self.arm = mlp(ARM_DIM, 32, 32)
            self.hand = mlp(HAND_DIM - 1, 64, 48)
            self.e_drop = nn.Dropout(0.3)
            self.e_body, self.e_face, self.e_hand = (nn.Linear(emb_dim, o) for o in (96, 48, 64))
            self.region = nn.Sequential(nn.Linear(48 + 64 + 2, 64), nn.GELU())
            self.trunk = nn.Sequential(nn.Linear(48 + 16 + 32 + 64 + 96 + 48 + 5, 160), nn.GELU(), nn.Dropout(0.25),
                                       nn.Linear(160, 64))
            self.out = nn.Linear(64, n_cls)
            for name, d in (("face", FACE_DIM - 1), ("pose", POSE_DIM - 1), ("arm", ARM_DIM), ("hand", HAND_DIM - 1)):
                self.register_buffer(f"mu_{name}", torch.zeros(d))
                self.register_buffer(f"sd_{name}", torch.ones(d))
            self.register_buffer("e_mu", torch.zeros(3, emb_dim))       # cuerpo, cara, mano
            self.register_buffer("e_scale", torch.ones(3))

        def norm(self, v, name):
            v = (v - getattr(self, f"mu_{name}")) / getattr(self, f"sd_{name}")
            return v + 0.15 * torch.randn_like(v) if self.training else v

        def streams(self, x, emb, ok):
            """x (B, LM_DIM), emb (B, N_CROPS, E), ok (B, N_CROPS) -> cada senal ya codificada, todas juntas"""
            B = x.shape[0]
            face, pose = x[:, O_FACE:O_POSE], x[:, O_POSE:O_ARMS]
            arms = x[:, O_ARMS:O_HANDS].reshape(B, 2, ARM_DIM)
            hands = x[:, O_HANDS:].reshape(B, 2, HAND_DIM)
            fp, pp, hp = face[:, :1], pose[:, :1], hands[:, :, :1]
            f = self.face(self.norm(face[:, 1:], "face")) * fp
            g = self.pose(self.norm(pose[:, 1:], "pose")) * pp
            a = self.arm(self.norm(arms, "arm")).sum(1) * pp
            hl = self.hand(self.norm(hands[:, :, 1:], "hand")) * hp
            if self.use_vision:
                kind = torch.tensor([0, 1, 2, 2], device=x.device)
                e = (emb - self.e_mu[kind]) * self.e_scale[kind, None]
                if self.training:
                    e = e + 0.25 * torch.randn_like(e)
                e = self.e_drop(e)
                eb, ef = self.e_body(e[:, 0]) * ok[:, 0:1], self.e_face(e[:, 1]) * ok[:, 1:2]
                eh, okh = self.e_hand(e[:, 2:]) * ok[:, 2:, None], ok[:, 2:, None]
            else:
                eb, ef = x.new_zeros(B, 96), x.new_zeros(B, 48)
                eh, okh = x.new_zeros(B, 2, 64), x.new_zeros(B, 2, 1)
            r = (self.region(torch.cat([hl, eh, hp, okh], -1)) * torch.maximum(hp, okh)).sum(1)
            flags = torch.cat([fp, pp, ok[:, 1:2] if self.use_vision else fp * 0, hp.sum(1) / 2, okh.sum(1) / 2], 1)
            return torch.cat([f, g, a, r, eb, ef, flags], 1)

        def forward(self, x, emb, ok, with_streams=False):
            h = self.streams(x, emb, ok)
            logits = self.out(F.gelu(self.trunk(h)))
            return (logits, F.normalize(h, dim=1)) if with_streams else logits

    return Net()


def _stats(v, floor=0.05):
    if len(v) < 2:
        return np.zeros(v.shape[1], np.float32), np.ones(v.shape[1], np.float32)
    return v.mean(0).astype(np.float32), np.maximum(v.std(0), floor).astype(np.float32)


class Classifier:
    def __init__(self, net, names, bank, scale, device):
        self.net, self.names, self.bank, self.scale, self.device = net, names, bank, scale, device
        self.torch = need("torch")
        self.use_vision = net.use_vision
        self.far = 0.0

    # ---- entrenamiento
    @classmethod
    def train(cls, data, names, use_vision, device, steps=TRAIN_STEPS, seed=0):
        torch = need("torch")
        F = torch.nn.functional
        torch.manual_seed(seed)
        X, y, OK = data["X"], data["y"], data["OK"]
        net = build_net(len(names), data["E"].shape[-1], use_vision)

        # normalizacion de cada grupo de features (solo con las muestras donde ese grupo existe)
        sd = net.state_dict()
        has_pose = X[:, O_POSE] > 0
        hands = X[:, O_HANDS:].reshape(-1, HAND_DIM)
        groups = {"face": X[X[:, O_FACE] > 0, O_FACE + 1:O_POSE], "pose": X[has_pose, O_POSE + 1:O_ARMS],
                  "arm": X[has_pose, O_ARMS:O_HANDS].reshape(-1, ARM_DIM), "hand": hands[hands[:, 0] > 0, 1:]}
        for k, v in groups.items():
            mu, s = _stats(v)
            sd[f"mu_{k}"], sd[f"sd_{k}"] = torch.from_numpy(mu), torch.from_numpy(s)
        if use_vision:
            for i, (e, ok) in enumerate(((data["E"][:, 0], OK[:, 0]), (data["E"][:, 1], OK[:, 1]),
                                         (data["E"][:, 2:].reshape(-1, data["E"].shape[-1]), OK[:, 2:].reshape(-1)))):
                e = e[ok > 0][::max(1, int((ok > 0).sum()) // 4000)].astype(np.float32)
                if len(e) > 1:
                    sd["e_mu"][i] = torch.from_numpy(e.mean(0))
                    sd["e_scale"][i] = 1.0 / max(float(e.std(0).mean()), 1e-4)
        net.load_state_dict(sd)
        net.to(device)

        # a cada clase le es asignado el mismo peso (a neutral un poco mas), y dentro de la clase a cada toma tambien
        w = np.zeros(len(y))
        for c in range(len(names)):
            tk = np.unique(data["take"][y == c])
            for t in tk:
                m = (y == c) & (data["take"] == t)
                w[m] = (NEUTRAL_WEIGHT if c == 0 else 1.0) / (len(tk) * m.sum())
        w = torch.from_numpy(w / w.sum()).float()
        Xt, yt, OKt = torch.from_numpy(X).to(device), torch.from_numpy(y).long().to(device), torch.from_numpy(OK).to(device)
        Et = torch.from_numpy(data["E"]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.02)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
        drop_cols = [O_FACE, O_POSE, O_HANDS, O_HANDS + HAND_DIM]
        net.train()
        for _ in range(steps):
            idx = torch.multinomial(w, 256, replacement=True).to(device)
            xb, ok = Xt[idx].clone(), OKt[idx].clone()
            # grupos enteros son apagados al azar: asi ninguna prediccion depende de una sola senal
            for c in drop_cols:
                xb[torch.rand(len(idx), device=device) < 0.1, c] = 0
            ok *= (torch.rand_like(ok) > 0.1).float()
            loss = F.cross_entropy(net(xb, Et[idx].float(), ok), yt[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()

        # banco de muestras para medir que tan lejos esta una pose nueva de lo grabado;
        # la escala de cada clase es la distancia tipica entre sus propias tomas
        b = data["base"]
        self = cls(net, names, None, None, device)
        z = self._forward(X[b], data["E"][b], OK[b])[1].cpu()
        yb, tb, fb = y[b], data["take"][b], data["frame"][b]
        rng = np.random.default_rng(seed)
        self.bank, self.scale = [], []
        for c in range(len(names)):
            rows = np.where(yb == c)[0]
            if len(rows) > 1500:
                rows = np.sort(rng.choice(rows, 1500, replace=False))
            zc, tc, fc = z[torch.from_numpy(rows)], tb[rows], fb[rows]
            D = torch.cdist(zc, zc)
            same = tc[:, None] == tc[None]
            if len(np.unique(tc)) < 2:      # una sola toma: son ignorados solo los cuadros vecinos en el tiempo
                same &= np.abs(fc[:, None] - fc[None]) <= max(2, int(fc.max() + 1) // 4)
            D[torch.from_numpy(same)] = float("inf")
            d = D.topk(min(KNN_K, len(zc)), largest=False).values
            d = torch.where(torch.isfinite(d), d, torch.nan).nanmean(1)
            d = d[torch.isfinite(d)]
            self.scale.append(float(d.median()) if len(d) else 0.0)
            self.bank.append(zc.to(device))
        floor = 0.7 * float(np.median([v for v in self.scale if v > 0] or [0.2]))
        self.scale = [max(v, floor) for v in self.scale]    # una clase con una sola toma no debe quedar mas estricta
        return self

    # ---- inferencia
    def _forward(self, X, E, OK):
        """-> (logits, senales codificadas y normalizadas)"""
        torch = self.torch
        with torch.inference_mode():
            return self.net(*(torch.from_numpy(np.asarray(v, np.float32)).to(self.device) for v in (X, E, OK)),
                            with_streams=True)

    def probs_batch(self, X, E, OK):
        """-> probabilidades de la red (B, C), probabilidades finales con rechazo (B, C), lejania (B,)"""
        torch = self.torch
        if not self.use_vision:
            E, OK = np.zeros((len(X), N_CROPS, 1), np.float32), np.zeros((len(X), N_CROPS), np.float32)
        logits, zn = self._forward(X, E, OK)
        with torch.inference_mode():
            raw = torch.softmax(logits, 1)
            best = raw.argmax(1)
            far = torch.zeros(len(X), device=zn.device)
            for c in best.unique().tolist():
                m = best == c
                k = min(KNN_K, len(self.bank[c]))
                far[m] = torch.cdist(zn[m], self.bank[c]).topk(k, largest=False).values.mean(1) / self.scale[c]
            keep = (1 - (far - REJECT_LO) / (REJECT_HI - REJECT_LO)).clamp(0, 1)
            keep[best == 0] = 1.0
            final = raw * keep[:, None]
            final[:, 0] += 1 - keep
        return raw.cpu().numpy(), final.cpu().numpy(), far.cpu().numpy()

    def probs(self, x, emb, ok):
        _, final, far = self.probs_batch(x[None], emb[None], ok[None])
        self.far = float(far[0])
        return final[0]

    # ---- guardar / cargar
    def save(self, signature):
        torch = self.torch
        torch.save({"state": self.net.state_dict(), "names": list(self.names), "signature": signature,
                    "use_vision": bool(self.use_vision), "emb_dim": int(self.net.e_mu.shape[1]),
                    "bank": [b.cpu() for b in self.bank], "scale": [float(s) for s in self.scale]}, MODEL_FILE)

    @classmethod
    def load(cls, signature, device):
        torch = need("torch")
        if not MODEL_FILE.exists():
            return None
        try:
            ck = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
            if ck["signature"] != signature:
                return None
            net = build_net(len(ck["names"]), ck["emb_dim"], ck["use_vision"])
            net.load_state_dict(ck["state"])
            return cls(net.eval().to(device), ck["names"], [b.to(device) for b in ck["bank"]], ck["scale"], device)
        except Exception:
            return None


def get_model(vision, device, n_var, force=False, quiet=False):
    """modelo.pt es cargado si corresponde a las tomas actuales; si no, el modelo es entrenado (toma segundos)."""
    names, takes = load_dataset(vision, n_var, quiet)
    sig = hashlib.sha1("|".join(
        [f"v{FEAT_VERSION}", str(vision is not None), str(TRAIN_STEPS), str(NEUTRAL_WEIGHT)]
        + [f"{t['name']}/{t['path'].name}:{t['key']}" for t in takes]).encode()).hexdigest()
    clf = None if force else Classifier.load(sig, device)
    if clf is None:
        if not quiet:
            print(f"Entrenando con {len(takes)} tomas de {len(names)} expresiones ...", flush=True)
        clf = Classifier.train(assemble(takes, names, vision is not None), names, vision is not None, device)
        clf.save(sig)
    if not quiet:
        count = {n: sum(len(t["x"]) for t in takes if t["name"] == n) for n in names}
        print("Modelo listo: " + ", ".join(f"{n}={count[n]}" for n in names))
    return clf


def setup(args):
    """-> (vision o None, device, versiones por cuadro)"""
    device = pick_device(args.device)
    vision = None if args.no_vision else Vision(device)
    if vision is not None and device == "cpu":
        print("Aviso: no hay GPU disponible para torch; el modelo de vision correra lento en CPU.\n"
              "       (si va muy lento, puede ser usado --no-vision, o torch puede ser instalado con CUDA)")
    return vision, device, (AUG_VARIANTS if device != "cpu" else 2)


# --------------------------------------------------------------------------
# Director: que meme esta activo + fundidos
# --------------------------------------------------------------------------
class Director:
    def __init__(self, n, delay=0.0):
        self.n, self.delay = n, delay
        self.sp = np.zeros(n)
        self.sp[0] = 1
        self.active, self.from_neutral = 0, False
        self.cand, self.cand_t, self.low_t = None, 0.0, None
        self.t_switch, self.t_obs = -1e9, None
        self.alpha = np.zeros(n)
        self.alpha[0] = 1

    def observe(self, p, t):
        """Es recibida una prediccion nueva (p) del cuadro capturado en el instante t."""
        dt = 0.05 if self.t_obs is None else float(np.clip(t - self.t_obs, 1e-3, 0.5))
        self.t_obs = t
        self.sp += (1 - math.exp(-dt / EMA_TAU)) * (p - self.sp)
        a = self.active
        if a != 0:                                  # el meme activo es soltado cuando ya no es sostenido
            if self.sp[a] >= OFF_MIN:
                self.low_t = None
            else:
                self.low_t = t if self.low_t is None else self.low_t
                if t - self.low_t >= RELEASE and t - self.t_switch >= MIN_SHOW:
                    self._switch(0, t)
        a = self.active
        best = int(np.argmax(self.sp[1:])) + 1 if self.n > 1 else 0
        if best and best != a and self.sp[best] >= ON_MIN and (a == 0 or self.sp[best] > self.sp[a] + MARGIN):
            if self.cand != best:
                self.cand, self.cand_t = best, t
            elif t - self.cand_t >= HOLD and t - self.t_switch >= (REARM if a == 0 else MIN_SHOW):
                self._switch(best, t)
        else:
            self.cand = None

    def _switch(self, c, t):
        self.from_neutral = self.active == 0
        self.active, self.t_switch, self.cand, self.low_t = c, t, None, None

    def animate(self, dt, now):
        """Los fundidos son avanzados (es llamado en cada cuadro que es dibujado)."""
        target = np.zeros(self.n)
        waiting = self.active != 0 and self.from_neutral and now - self.t_switch < self.delay
        if not waiting:
            target[self.active] = 1
        d = target - self.alpha
        self.alpha += np.where(d > 0, np.minimum(d, FADE_IN * dt), -np.minimum(-d, FADE_OUT * dt))


def render(frame, layers, director, mode, pip, bg):
    alpha, active = director.alpha, director.active
    base = frame if mode == 1 else np.full((SQ, SQ, 3), bg, np.uint8)
    order = sorted((c for c in layers if alpha[c] > 0.004), key=lambda c: (c == active, alpha[c]))
    top = layers[order[-1]] if order else None
    if top is not None and alpha[order[-1]] > 0.996 and top[2] is not None:
        x0, y0, x1, y1 = top[2]                     # meme ya opaco y sin transparencia: es pegado directo
        out = base.copy()
        out[y0:y1, x0:x1] = top[0][y0:y1, x0:x1]
    elif order:
        out = base.astype(np.float32)
        for c in order:                             # el meme activo es dibujado al final (encima)
            img, mask, _ = layers[c]
            a = float(alpha[c])
            out += (img - out) * (mask * (a * a * (3 - 2 * a)))
        out = out.astype(np.uint8)
    else:
        out = base.copy()
    if mode == 2 and pip:
        h = int(160 * frame.shape[0] / frame.shape[1])
        out[8:8 + h, 8:8 + 160] = cv2.resize(frame, (160, h))
    return out


# --------------------------------------------------------------------------
# Ventana
# --------------------------------------------------------------------------
def screen_size():
    """(borde derecho, borde inferior) de la pantalla principal, en pixeles."""
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        pass
    try:
        import re
        import subprocess
        out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=2).stdout
        m = (re.search(r" connected primary (\d+)x(\d+)\+(\d+)\+(\d+)", out)
             or re.search(r" connected (\d+)x(\d+)\+(\d+)\+(\d+)", out))
        if m:
            w, h, x, y = map(int, m.groups())
            return x + w, y + h
    except Exception:
        pass
    try:
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def layout(win, size, mini):
    w, h = size
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    except cv2.error:
        pass
    if mini:
        s = 360 / max(w, h)
        tw, th = int(w * s), int(h * s)
        sw, sh = screen_size()
        cv2.resizeWindow(win, tw, th)
        cv2.moveWindow(win, sw - tw - 20, sh - th - 80)
    else:
        cv2.resizeWindow(win, w, h)
        cv2.moveWindow(win, 100, 100)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1 if mini else 0)
    except cv2.error:
        if mini:
            print("Aviso: este OpenCV no soporta 'siempre encima'.")


# --------------------------------------------------------------------------
# CALIBRAR
# --------------------------------------------------------------------------
TIPS_NEUTRAL = ["habla y riete normal", "mira a los lados y hacia abajo", "mueve las manos: mouse, teclado, control",
                "tocate la cara, rascate, acomodate", "acercate y alejate de la camara",
                "manos vacias cerca de la cara y del pecho"]
TIPS_POSE = ["manten la pose", "acercate un poco", "alejate un poco", "gira levemente", "muevete un poco sin soltar la pose"]


def calibrate(args):
    cam = Camera(args.cam)
    W, H = cam.W, cam.H
    memes = load_memes(W, H)
    names = ["neutral"] + sorted(n for n in memes if n != "neutral")
    if args.only:
        unknown = [n for n in args.only if n not in names]
        if unknown:
            sys.exit(f"No existe en la carpeta memes/: {', '.join(unknown)}\nDisponibles: {', '.join(names)}")
        names = [n for n in names if n in args.only]
    ex = Extractor()
    win = "Calibracion"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win, W, H)

    def hud(view, o):
        draw_obs(view, o)
        put_text(view, f"cara: {'SI' if o.has_face else 'NO'}   manos: {o.n_hands}   cuerpo: {'SI' if o.has_pose else 'NO'}",
                 (30, H - 55), 0.6, (0, 255, 0) if o.has_face else (0, 0, 255))

    def record(name):
        seconds = args.seconds or (REC_SECONDS_NEUTRAL if name == "neutral" else REC_SECONDS)
        fps = float(np.clip(240 / seconds, 3, STORE_FPS))
        tips = TIPS_NEUTRAL if name == "neutral" else TIPS_POSE
        t0 = time.time()
        while time.time() - t0 < REC_COUNTDOWN:
            view = cam.latest()[1].copy()
            put_text(view, f"{name}: prepara la pose... {math.ceil(REC_COUNTDOWN - (time.time() - t0))}",
                     (30, 60), 1.1, (0, 255, 255))
            cv2.imshow(win, view)
            cv2.waitKey(15)
        writer = TakeWriter(name, fps, (W, H))
        t0, last, cancel = time.time(), -1, False
        while not cancel and not cam.ended:
            el = time.time() - t0
            if el >= seconds:
                break
            idx, frame = cam.latest()
            if idx != last and el >= writer.n / fps:
                writer.write(frame)
                last = idx
            view = frame.copy()
            hud(view, ex(frame))
            put_text(view, f"GRABANDO {name}  {seconds - el:.0f}s", (30, 50), 1.0, (0, 0, 255))
            put_text(view, tips[int(el / max(2.5, seconds / len(tips))) % len(tips)], (30, 90), 0.75, (0, 255, 255))
            cv2.rectangle(view, (0, H - 12), (int(W * el / seconds), H), (0, 255, 0), -1)
            cv2.imshow(win, view)
            cancel = cv2.waitKey(1) & 0xFF == 27
        path = writer.close()
        if cancel or writer.n < 5:
            delete_take(path)
            print(f"Toma de {name} cancelada.")
        else:
            print(f"Toma guardada: {name}/{path.name} ({writer.n} cuadros)")

    i = 0
    while 0 <= i < len(names) and not cam.ended:
        name = names[i]
        view = cam.latest()[1].copy()
        hud(view, ex(view))
        takes = takes_of(name)
        put_text(view, f"[{i + 1}/{len(names)}] {name.upper()}   tomas: {len(takes)}", (30, 50), 1.0, (0, 255, 255))
        tip = ("NEUTRAL = lo que NO es meme (jugar, hablar, reir). Son grabadas varias tomas."
               if name == "neutral" else "Son grabadas 2-3 tomas (ESPACIO), variando distancia y angulo")
        put_text(view, tip, (30, 85), 0.55, (0, 255, 0))
        put_text(view, "ESPACIO=grabar toma  X=borrar ultima toma  ENTER=siguiente  B=anterior  Q=terminar",
                 (30, H - 22), 0.55)
        if name in memes:
            ref = cv2.resize(memes[name][0].astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            view[105:105 + ref.shape[0], W - ref.shape[1] - 10:W - 10] = ref
        cv2.imshow(win, view)
        k = cv2.waitKey(15) & 0xFF
        if k == ord(' '):
            record(name)
        elif k in (ord('x'), ord('X'), 8, 127) and takes:
            delete_take(takes[-1])
            print(f"Borrada la ultima toma de {name}.")
        elif k in (13, 10):
            i += 1
        elif k in (ord('b'), ord('B')):
            i = max(0, i - 1)
        elif k in (ord('q'), ord('Q'), 27):
            break
    cam.close()
    ex.close()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if "neutral" not in dataset_names():
        sys.exit("'neutral' no ha sido grabado (la cara normal). calibrate debe ser ejecutado de nuevo.")
    if len(dataset_names()) < 2:
        sys.exit("Neutral ya fue grabado; ahora debe ser grabada al menos una expresion con calibrate.")
    print("\nLas tomas estan siendo procesadas y el modelo esta siendo entrenado (la primera vez tarda mas)...")
    vision, device, n_var = setup(args)
    get_model(vision, device, n_var)
    print("\nListo. Puede ser probado con:  python memes_webcam.py test   y luego   python memes_webcam.py run")


# --------------------------------------------------------------------------
# TRAIN / TEST
# --------------------------------------------------------------------------
def train(args):
    vision, device, n_var = setup(args)
    get_model(vision, device, n_var, force=True)


def test(args):
    """Se muestra que poses son confundidas entre si; la evaluacion siempre es hecha con tomas no vistas por el modelo al entrenar."""
    vision, device, n_var = setup(args)
    names, takes = load_dataset(vision, n_var)
    data = assemble(takes, names, vision is not None)
    K, n = 3, len(names)
    fold = np.zeros(len(data["y"]), int)
    for c in range(n):
        tk = np.unique(data["take"][data["y"] == c])
        for j, t in enumerate(tk):
            m = data["take"] == t
            if len(tk) >= 2:
                fold[m] = j % K                                       # toma completa fuera
            else:
                fold[m] = np.minimum(data["frame"][m] * K // (data["frame"][m].max() + 1), K - 1)   # bloques de tiempo
    conf, act = np.zeros((n, n), int), np.zeros(n, int)
    print("Evaluando (puede tardar)...", flush=True)
    for k in range(K):
        te = (fold == k) & data["base"] & ~data["flip"]       # es evaluado solo con cuadros originales
        if not te.any():
            continue
        clf = Classifier.train(subset(data, fold != k), names, vision is not None, device, seed=k)
        raw, final, _ = clf.probs_batch(data["X"][te], data["E"][te], data["OK"][te])
        yt = data["y"][te]
        np.add.at(conf, (yt, raw.argmax(1)), 1)
        best = final.argmax(1)
        fire = (best != 0) & (final[np.arange(len(best)), best] >= ON_MIN)
        np.add.at(act, yt, np.where(yt == 0, fire, fire & (best == yt)))
    tot = np.maximum(1, conf.sum(1))
    print(f"\nfilas = lo que fue hecho, columnas = lo que fue predicho   ({'con' if vision else 'SIN'} modelo de vision)\n")
    print(" " * 16 + " ".join(f"{nm[:7]:>7}" for nm in names))
    for i, nm in enumerate(names):
        print(f"{nm[:15]:<16}" + " ".join(f"{conf[i, j]:>7}" for j in range(n)))
    print()
    for i, nm in enumerate(names):
        extra = (f"activaciones falsas {100 * act[i] / tot[i]:5.1f}%" if i == 0 else
                 f"se activaria {100 * act[i] / tot[i]:5.1f}%")
        print(f"{nm:<16} acierto {100 * conf[i, i] / tot[i]:5.1f}%   {extra}   ({len(takes_of(nm))} tomas)")
    print("\nConfusiones principales:")
    found = False
    for i in range(n):
        for j in range(n):
            if i != j and conf[i, j] / tot[i] > 0.10:
                found = True
                print(f"  {names[i]} es confundido con {names[j]} ({100 * conf[i, j] / tot[i]:.0f}%)")
    if not found:
        print("  ninguna importante")
    print("\nConsejos:")
    print("  - Poco acierto en una pose: deben ser grabadas mas tomas de esa pose con variedad (o debe ser hecha mas distinta).")
    print("  - Otras poses son absorbidas por 'neutral': esas poses son muy suaves y deben ser exageradas un poco.")
    print("  - Activaciones falsas altas: debe ser grabado mas neutral (calibrate --only neutral), parecido a como se juega.")
    print("  - 'se activaria' bajo con acierto alto: deben ser aumentados REJECT_LO / REJECT_HI o reducido ON_MIN.")


# --------------------------------------------------------------------------
# JUGAR
# --------------------------------------------------------------------------
class Perception(threading.Thread):
    """El cuadro mas reciente es analizado en un hilo aparte: la ventana nunca es trabada por el analisis."""

    def __init__(self, cam, ex, vision, clf, max_fps):
        super().__init__(daemon=True)
        self.cam, self.ex, self.vision, self.clf, self.max_fps = cam, ex, vision, clf, max_fps
        self.result = None          # (numero, instante, probabilidades, Obs, lejania, seg. de analisis)
        self.ring = deque(maxlen=int((FP_SECONDS + 1) * STORE_FPS))
        self.error, self.alive = None, True

    def run(self):
        last, seq, t_ring = -1, 0, 0.0
        no_e, no_ok = np.zeros((N_CROPS, 1), np.float32), np.zeros(N_CROPS, np.float32)
        try:
            while self.alive:
                idx, frame = self.cam.latest()
                if idx == last:
                    time.sleep(0.003)
                    continue
                last, t0 = idx, time.time()
                o = self.ex(frame)
                emb, ok = self.vision.embed_boxes(frame, o.boxes) if self.vision else (no_e, no_ok)
                clf = self.clf
                p = clf.probs(o.x, emb, ok)
                seq += 1
                self.result = (seq, t0, p, o, clf.far, time.time() - t0)
                if t0 - t_ring >= 1 / STORE_FPS:
                    self.ring.append((t0, frame))
                    t_ring = t0
                time.sleep(max(0.0, 1 / self.max_fps - (time.time() - t0)))
        except Exception as e:      # asi la ventana no es dejada congelada sin explicacion por un error del analisis
            self.error = e


def run(args):
    vision, device, n_var = setup(args)
    clf = get_model(vision, device, n_var)
    names = clf.names
    cam = Camera(args.cam)
    W, H = cam.W, cam.H
    memes_w, memes_s = load_memes(W, H), load_memes(SQ, SQ)
    by_w = {i: memes_w[n] for i, n in enumerate(names) if n in memes_w and n != "neutral"}
    by_s = {i: memes_s[n] for i, n in enumerate(names) if n in memes_s}
    missing = [n for n in memes_w if n not in names]
    if missing:
        print("Aviso: sin calibrar (no se activaran):", ", ".join(missing))
    no_img = [n for n in names[1:] if n not in memes_w]
    if no_img:
        print("Aviso: grabadas pero sin imagen en memes/ (se detectan pero no muestran nada):", ", ".join(no_img))

    worker = Perception(cam, Extractor(), vision, clf, args.fps)
    worker.start()
    print("Teclas: 1/2 = modo | p = mini-webcam | d = debug | n = eso NO era un meme | t = mini | f = pantalla completa | q = salir")
    mode, pip, debug, full, mini = 1, True, False, False, args.mini
    if args.solo:
        mode, pip = 2, False
    director = Director(len(names), OVERLAY_DELAY if mode == 1 else 0.0)
    bg = BG_COLORS[args.bg]
    win = "Memes"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    size = lambda: (W, H) if mode == 1 else (SQ, SQ)
    layout(win, size(), mini)
    status, status_t, busy = "", 0.0, [False]

    def learn_neutral():
        """Tecla n: lo ultimo ocurrido NO era un meme -> es guardado como neutral y el modelo es reentrenado."""
        frames = [f for t, f in list(worker.ring) if time.time() - t <= FP_SECONDS]
        if len(frames) < 5 or busy[0]:
            return "un momento, por favor..."
        busy[0] = True
        writer = TakeWriter("neutral", STORE_FPS, (W, H), prefix="fp-")
        for f in frames:
            writer.write(f)
        path = writer.close()
        print(f"Guardado como neutral: {path}  (puede ser borrado si fue un error)")

        def job():
            nonlocal status, status_t
            try:
                new = get_model(vision, device, n_var, quiet=True)
                if new.names == worker.clf.names:
                    worker.clf = new
                status = "aprendido: eso era neutral"
            except BaseException as e:
                status = f"no pudo ser reentrenado: {e}"
            status_t, busy[0] = time.time(), False

        threading.Thread(target=job, daemon=True).start()
        return "esta siendo aprendido que eso era neutral..."

    last_seq, t_prev = 0, time.time()
    while not cam.ended:
        t_loop = time.time()
        if worker.error is not None:
            print("Error en el analisis:", repr(worker.error))
            break
        _, frame = cam.latest()
        res = worker.result
        if res is not None and res[0] != last_seq:
            last_seq = res[0]
            director.observe(res[2], res[1])
        director.delay = OVERLAY_DELAY if mode == 1 else 0.0
        director.animate(min(0.2, t_loop - t_prev), t_loop)
        t_prev = t_loop

        out = render(frame, by_w if mode == 1 else by_s, director, mode, pip, bg)
        if debug and res is not None:
            o = res[3]
            if mode == 1:
                draw_obs(out, o)
            oh = out.shape[0]
            put_text(out, f"activo:{names[director.active]}  analisis:{1000 * res[5]:.0f}ms  cara:{'SI' if o.has_face else 'NO'} "
                          f"manos:{o.n_hands} cuerpo:{'SI' if o.has_pose else 'NO'}  lejania:{res[4]:.1f}",
                     (10, oh - 12), 0.45, thick=1)
            for i, nm in enumerate(names):
                y0 = 110 + i * 20
                cv2.rectangle(out, (10, y0), (10 + int(160 * director.sp[i]), y0 + 14),
                              (0, 255, 0) if i == director.active else (0, 180, 255), -1)
                put_text(out, nm, (180, y0 + 13), 0.45, thick=1)
            cv2.line(out, (10 + int(160 * ON_MIN), 106), (10 + int(160 * ON_MIN), 110 + len(names) * 20), (255, 255, 255), 1)
        if status and (busy[0] or time.time() - status_t < 3):
            put_text(out, status, (10, 30), 0.6, (0, 255, 255))
        cv2.imshow(win, out)
        k = cv2.waitKey(max(1, int(33 - 1000 * (time.time() - t_loop)))) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k in (ord('1'), ord('2')):
            mode = 1 if k == ord('1') else 2
            layout(win, size(), mini)
        elif k == ord('p'):
            pip = not pip
        elif k == ord('d'):
            debug = not debug
        elif k == ord('n'):
            status, status_t = learn_neutral(), time.time()
        elif k == ord('t'):
            mini, full = not mini, False
            layout(win, size(), mini)
        elif k == ord('f'):
            if mini:
                mini = False
                layout(win, size(), mini)
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    if cam.ended and not cam.is_file:
        print("La camara dejo de entregar imagen.")
    worker.alive = False
    worker.join(timeout=2)
    cam.close()
    worker.ex.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Memes por expresion con webcam")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "calibrate", "test", "train"])
    ap.add_argument("--cam", default="0", help="indice de la camara (0, 1, 10) o ruta (/dev/video10)")
    ap.add_argument("--only", nargs="*", help="(calibrate) solo estas expresiones")
    ap.add_argument("--seconds", type=float, help="(calibrate) duracion de cada toma en segundos")
    ap.add_argument("--solo", action="store_true", help="(run) solo memes, ventana cuadrada, sin webcam")
    ap.add_argument("--mini", action="store_true", help="(run) ventana pequena en la esquina")
    ap.add_argument("--bg", default="negro", choices=list(BG_COLORS), help="(run) color de fondo del modo solo memes")
    ap.add_argument("--fps", type=float, default=MAX_FPS, help="(run) analisis por segundo como maximo")
    ap.add_argument("--no-vision", action="store_true", help="sin modelo de vision: solo cara, manos y brazos (mas ligero)")
    ap.add_argument("--device", default="auto", help="auto, cuda o cpu")
    a = ap.parse_args()
    {"calibrate": calibrate, "run": run, "test": test, "train": train}[a.command](a)
