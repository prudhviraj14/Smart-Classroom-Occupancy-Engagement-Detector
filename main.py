# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # ======================
# # Load YOLO for Face Detection
# # ======================
# yolo_net = cv2.dnn.readNetFromDarknet("yolov3-face.cfg", "yolov3-face.weights")
# yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# output_layer_names = yolo_net.getUnconnectedOutLayersNames()

# # ======================
# # Load Emotion Model (MobileNetV2)
# # ======================
# emotion_model = load_model("best_emotion_model.h5")
# emotion_labels = ["Angry", "Fear","Happy", "Neutral", "Sad", "Sleep","Surprise"]

# # ======================
# # Preprocessing Function
# # ======================
# def preprocess_face_48_rgb(crop_bgr):
#     face = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_AREA)
#     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # BGR → RGB
#     face = face.astype(np.float32)
#     face = preprocess_input(face)  # MobileNetV2 normalization
#     face = np.expand_dims(face, axis=0)  # (1,48,48,3)
#     return face

# # ======================
# # Start Webcam
# # ======================
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     h, w = frame.shape[:2]

#     # YOLO face detection
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
#     yolo_net.setInput(blob)
#     detections = yolo_net.forward(output_layer_names)

#     boxes = []
#     confidences = []
#     for detection in detections:
#         for obj in detection:
#             scores = obj[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:  # threshold
#                 box = obj[0:4] * np.array([w, h, w, h])
#                 (centerX, centerY, bw, bh) = box.astype("int")

#                 x = int(centerX - bw / 2)
#                 y = int(centerY - bh / 2)

#                 boxes.append([x, y, int(bw), int(bh)])
#                 confidences.append(float(confidence))

#     # Non-max suppression
#     idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

#     if len(idxs) > 0:
#         for i in idxs.flatten():
#             x, y, bw, bh = boxes[i]
#             x, y = max(0, x), max(0, y)
#             crop = frame[y:y+bh, x:x+bw]

#             if crop.size == 0:
#                 continue

#             # Preprocess face
#             inp = preprocess_face_48_rgb(crop)

#             # Predict emotion
#             preds = emotion_model.predict(inp, verbose=0)
#             emotion = emotion_labels[np.argmax(preds)]

#             # Draw results
#             cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
#             cv2.putText(frame, emotion, (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#     cv2.imshow("Classroom Engagement Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import os
import time
import cv2
import numpy as np
from collections import deque

# ========= CONFIG =========
# Path to your trained model (48x48 grayscale). Must match labels below.
EMOTION_MODEL_PATH = "emotion_model_7class.h5"

# If your model has 7 classes including Sleep, keep this order the same as training.
EMOTION_LABELS = ["Angry", "Happy", "Neutral", "Sad","Sleep"]
# If your model has only 6 classes (no Sleep), use:
# EMOTION_LABELS = ["Angry", "Happy", "Sad", "Fear", "Surprise", "Neutral"]

# YOLO weights (Ultralytics auto-downloads if missing)
YOLO_PERSON_WEIGHTS = "yolov8n.pt"
YOLO_FACE_WEIGHTS   = "yolov8n-face.pt"  # get from Ultralytics face model zoo

# Heuristics / thresholds
NO_FACE_THRESHOLD_FRAMES = 12     # ~0.4s @ 30 FPS: mark as HeadDown when no face in head region
SAD_ALERT_THRESHOLD       = 5      # >5 sad students -> alert
SLEEP_ALERT_THRESHOLD     = 1      # >=1 Sleep/HeadDown -> alert
ALERT_COOLDOWN_SEC        = 3.0    # don't beep more than once every N seconds

# Inference throttling (speed-up)
EMOTION_EVERY_N_FRAMES    = 2      # only run emotion on a track every N frames

DRAW_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ========= HEAVY IMPORTS AFTER CONFIG =========
from ultralytics import YOLO
from tensorflow.keras.models import load_model


# ========= SIMPLE IOU TRACKER =========
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0

class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.last_seen = time.time()
        self.no_face_streak = 0
        self.emotion_hist = deque(maxlen=8)
        self.current_emotion = None
        self._frame_count = 0  # throttle emotion inference

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_lost_time=2.0):
        self.iou_thresh = iou_thresh
        self.max_lost_time = max_lost_time
        self.tracks = []
        self.next_id = 1

    def update(self, dets):
        now = time.time()
        assigned = set()
        # match existing tracks
        for t in self.tracks:
            best_iou, best_j = 0.0, -1
            for j, d in enumerate(dets):
                if j in assigned: 
                    continue
                sc = iou(t.box, d)
                if sc > best_iou:
                    best_iou, best_j = sc, j
            if best_iou >= self.iou_thresh and best_j != -1:
                t.box = dets[best_j]
                t.last_seen = now
                assigned.add(best_j)
        # new tracks
        for j, d in enumerate(dets):
            if j not in assigned:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1
        # prune old
        self.tracks = [t for t in self.tracks if (now - t.last_seen) <= self.max_lost_time]
        return self.tracks


# ========= LOAD MODELS =========
print("[*] Loading YOLO models...")
yolo_person = YOLO(YOLO_PERSON_WEIGHTS)
yolo_face   = YOLO(YOLO_FACE_WEIGHTS)

print("[*] Loading emotion model...")
emotion_model = load_model(EMOTION_MODEL_PATH)
num_out = emotion_model.output_shape[-1]
if num_out != len(EMOTION_LABELS):
    raise ValueError(
        f"Model outputs {num_out} classes but EMOTION_LABELS has {len(EMOTION_LABELS)}. "
        "Fix EMOTION_LABELS order to match training."
    )

# ========= HELPERS =========
def preprocess_face_48_gray(gray_crop):
    # For your 48x48 grayscale model
    face = cv2.resize(gray_crop, (48, 48), interpolation=cv2.INTER_AREA)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1,48,48,1)
    return face

def majority(votes):
    if not votes:
        return None
    vals, counts = np.unique(votes, return_counts=True)
    return vals[np.argmax(counts)]

def head_region(box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    # top 35% as "head region"
    return [x1, y1, x2, int(y1 + 0.35 * h)]

def overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a

def draw_box(img, box, text, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, DRAW_THICKNESS)
    if text:
        cv2.putText(img, text, (x1, max(20, y1 - 8)), FONT, 0.6, color, 2, cv2.LINE_AA)

def center_of(box):
    x1, y1, x2, y2 = box
    return ( (x1+x2)//2, (y1+y2)//2 )

_last_alert_time = 0.0
def alert_once(msg):
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time < ALERT_COOLDOWN_SEC:
        return
    _last_alert_time = now
    print("[ALERT]", msg)
    try:
        import winsound
        winsound.Beep(1000, 350)
        winsound.Beep(1400, 350)
    except Exception:
        # Fallback: no sound available
        pass


# ========= MAIN =========
def main():
    cap = cv2.VideoCapture(0)  # set to a video path to test prerecorded footage
    if not cap.isOpened():
        print("ERROR: Cannot open camera/video.")
        return

    tracker = SimpleTracker()

    print("[*] Running… Press 'q' to quit.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Person detection
        person_boxes = []
        for r in yolo_person(frame):
            if r.boxes is None: 
                continue
            for b in r.boxes:
                if int(b.cls[0].item()) == 0:  # 'person'
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    x1, y1 = max(0,x1), max(0,y1)
                    x2, y2 = min(w-1,x2), min(h-1,y2)
                    if x2 > x1 and y2 > y1:
                        person_boxes.append([x1,y1,x2,y2])

        tracks = tracker.update(person_boxes)

        # 2) Face detection once per frame
        face_boxes = []
        for r in yolo_face(frame):
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(w-1,x2), min(h-1,y2)
                if x2 > x1 and y2 > y1:
                    face_boxes.append([x1,y1,x2,y2])

        sad_count = 0
        sleep_like_count = 0

        for t in tracks:
            t._frame_count += 1
            # look for the face inside the head region of this person
            head_box = head_region(t.box)
            best_face = None
            best_ov = 0.0
            for fb in face_boxes:
                ov = overlap_ratio(fb, head_box)
                if ov > best_ov:
                    best_ov = ov
                    best_face = fb

            if best_face is not None and best_ov > 0.12:
                # face found for this track
                fx1, fy1, fx2, fy2 = best_face
                crop = gray[fy1:fy2, fx1:fx2]
                if crop.size > 0 and (t._frame_count % EMOTION_EVERY_N_FRAMES == 0):
                    inp = preprocess_face_48_gray(crop)
                    preds = emotion_model.predict(inp, verbose=0)[0]
                    lab = EMOTION_LABELS[int(np.argmax(preds))]
                    t.emotion_hist.append(lab)
                t.current_emotion = majority(list(t.emotion_hist)) or t.current_emotion

                t.no_face_streak = 0
                draw_box(frame, t.box, f"ID {t.id}", (0,255,0))
                draw_box(frame, best_face, t.current_emotion or "…", (255,0,0))
            else:
                # no face seen in head region
                t.no_face_streak += 1
                draw_box(frame, t.box, f"ID {t.id}", (0,255,255))
                if t.no_face_streak >= NO_FACE_THRESHOLD_FRAMES:
                    # treat as head-down / sleep-like
                    t.current_emotion = "Sleep" if "Sleep" in EMOTION_LABELS else "HeadDown"
                    hx1, hy1, hx2, hy2 = head_box
                    cv2.rectangle(frame, (hx1,hy1), (hx2,hy2), (0,165,255), 2)
                    cv2.putText(frame, "Head Down", (hx1, max(20, hy1-8)), FONT, 0.55, (0,165,255), 2)

            # aggregates
            if t.current_emotion == "Sad":
                sad_count += 1
            if t.current_emotion in ("Sleep", "HeadDown"):
                sleep_like_count += 1

            # overlay current emotion near the person center
            if t.current_emotion:
                cx, cy = center_of(t.box)
                cv2.putText(frame, t.current_emotion, (cx-40, cy),
                            FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # 3) Top summary & alerts
        student_count = len(tracks)
        cv2.putText(frame, f"Students: {student_count}", (12, 30), FONT, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"Sad: {sad_count}", (12, 60), FONT, 0.8, (0,200,255), 2)
        cv2.putText(frame, f"Sleep/HeadDown: {sleep_like_count}", (12, 90), FONT, 0.8, (0,165,255), 2)

        banner = None
        if sleep_like_count >= SLEEP_ALERT_THRESHOLD:
            banner = "ALERT: Student Sleep / Head-Down detected!"
        elif sad_count > SAD_ALERT_THRESHOLD:
            banner = f"ALERT: Sad students > {SAD_ALERT_THRESHOLD}!"

        if banner:
            cv2.rectangle(frame, (0, h-42), (w, h), (0,0,255), -1)
            cv2.putText(frame, banner, (14, h-12), FONT, 0.8, (255,255,255), 2, cv2.LINE_AA)
            alert_once(banner)

        cv2.imshow("Smart Classroom Monitor", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"ERROR: '{EMOTION_MODEL_PATH}' not found. Set EMOTION_MODEL_PATH correctly.")
    else:
        main()
