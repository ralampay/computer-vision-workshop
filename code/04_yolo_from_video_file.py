import cv2
import torch
from ultralytics import YOLO

# 1) Load a pretrained YOLOv11 model (auto-downloads on first run)
# Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
model = YOLO("yolo11n.pt")

# If NVIDIA cuda is available, load it
if torch.cuda.is_available():
    model.to("cuda")

# 2) Helper to draw a box with a label background
def draw_box_with_label(frame, x1, y1, x2, y2, label, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, text_th = 0.6, 2
    (tw, th), base = cv2.getTextSize(label, font, font_scale, text_th)
    bg_x1, bg_y1 = x1, max(0, y1 - th - base - 6)
    bg_x2, bg_y2 = x1 + tw + 8, y1
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 4, y1 - base - 3), font, font_scale, (255, 255, 255), text_th, cv2.LINE_AA)

# 3) Open the video file
cap = cv2.VideoCapture("example-video.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open example-video.mp4")

names = model.names  # class index -> class name

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4) Run YOLO inference on the BGR frame
    result = model(frame, conf=0.25, verbose=False)[0]

    # 5) Parse detections and draw
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        conf = result.boxes.conf.detach().cpu().numpy()
        cls  = result.boxes.cls.detach().cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            class_name = names[int(k)] if isinstance(names, (list, dict)) else str(k)
            label = f"{class_name} {c:.2f}"
            # simple per-class color
            color = (int(37 * (k + 1) % 255), int(17 * (k + 1) % 255), int(29 * (k + 1) % 255))
            draw_box_with_label(frame, x1, y1, x2, y2, label, color=color, thickness=2)

    # 6) Show frame
    cv2.imshow("YOLOv11 Video", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
