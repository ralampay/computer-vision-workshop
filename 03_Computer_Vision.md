# Computer Vision

## Draw and Detect on Frame

```python
import cv2
import numpy as np
from ultralytics import YOLO

# 1) Load a pretrained YOLOv11 model (auto-downloads on first run)
# Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
model = YOLO("yolo11n.pt")

# 2) Read image
image_path = "example.jpg"  # change this
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# 3) Run inference (returns a list; take the first result)
# You can adjust conf (confidence threshold) as needed, e.g., conf=0.25
result = model(img, conf=0.25, verbose=False)[0]

# 4) Names map for class indices
names = model.names  # dict or list mapping class id -> name

# 5) Helper: draw rectangle + label with background
def draw_box_with_label(frame, x1, y1, x2, y2, label, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle and a filled label background above it.
    frame: BGR image (OpenCV)
    (x1, y1): top-left corner
    (x2, y2): bottom-right corner
    label: string to render (e.g., 'person 0.90')
    """
    # rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # compute text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_th = 2
    (tw, th), base = cv2.getTextSize(label, font, font_scale, text_th)

    # background box (placed above top-left, clamp if off-frame)
    bg_x1, bg_y1 = x1, max(0, y1 - th - base - 6)
    bg_x2, bg_y2 = x1 + tw + 8, y1

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # filled black bg
    cv2.putText(frame, label, (x1 + 4, y1 - base - 3), font, font_scale, (255, 255, 255), text_th, cv2.LINE_AA)

# 6) Extract detections and draw
# result.boxes contains:
#  - .xyxy (N,4) tensor of [x1,y1,x2,y2]
#  - .conf (N,) tensor of confidences
#  - .cls  (N,) tensor of class indices
if result.boxes is not None and len(result.boxes) > 0:
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    cls  = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        class_name = names[int(k)] if isinstance(names, (list, dict)) else str(k)
        label = f"{class_name} {c:.2f}"
        # Choose color per class (simple hash)
        color = (int(37 * (k + 1) % 255), int(17 * (k + 1) % 255), int(29 * (k + 1) % 255))
        draw_box_with_label(img, x1, y1, x2, y2, label, color=color, thickness=2)
else:
    print("No detections found.")

# 7) Show in a window
cv2.imshow("YOLOv11 Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Draw and Detect on Webcam

```python
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

# 1) Load a pretrained YOLOv11 model (auto-downloads on first run)
# Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
model = YOLO("yolo11n.pt")

# (Optional) Send model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

# 2) Helper: draw rectangle + label with background
def draw_box_with_label(frame, x1, y1, x2, y2, label, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_th = 2
    (tw, th), base = cv2.getTextSize(label, font, font_scale, text_th)

    # background box (placed above top-left, clamp if off-frame)
    bg_x1, bg_y1 = x1, max(0, y1 - th - base - 6)
    bg_x2, bg_y2 = x1 + tw + 8, y1

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # filled black bg
    cv2.putText(frame, label, (x1 + 4, y1 - base - 3), font, font_scale, (255, 255, 255), text_th, cv2.LINE_AA)

# 3) Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# (Optional) FPS measurement
prev_time = time.time()
frame_counter = 0
fps_display = "FPS: --"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4) Run YOLO inference on the BGR frame
    # Tip: adjust conf=0.25..0.5 to filter low-confidence boxes
    result = model(frame, conf=0.25, verbose=False)[0]

    # 5) Parse detections and draw
    names = model.names  # class index -> name

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls  = result.boxes.cls

        # Move tensors to CPU and convert to numpy
        xyxy = xyxy.detach().cpu().numpy().astype(int)
        conf = conf.detach().cpu().numpy()
        cls  = cls.detach().cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            class_name = names[int(k)] if isinstance(names, (list, dict)) else str(k)
            label = f"{class_name} {c:.2f}"
            # Simple per-class color
            color = (int(37 * (k + 1) % 255), int(17 * (k + 1) % 255), int(29 * (k + 1) % 255))
            draw_box_with_label(frame, x1, y1, x2, y2, label, color=color, thickness=2)

    # (Optional) Update and overlay FPS every ~0.5s
    frame_counter += 1
    now = time.time()
    if now - prev_time >= 0.5:
        fps = frame_counter / (now - prev_time)
        fps_display = f"FPS: {fps:.1f}"
        frame_counter = 0
        prev_time = now

    cv2.putText(frame, fps_display, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 6) Show frame
    cv2.imshow("YOLOv11 Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
