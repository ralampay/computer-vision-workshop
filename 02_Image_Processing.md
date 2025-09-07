# Image Processing

## Draw a Rectangle

* `(x1, y1) = (50, 60)` is the top-left corner.
* `(x2, y2) = (300, 240)` is the bottom-right corner.
* The rectangle spans `width = x2 - x1 = 250` and `height = y2 - y1 = 180` pixels.

```python
import cv2

img = cv2.imread("example.jpg")

# Define corners
x1, y1 = 50, 60      # top-left
x2, y2 = 300, 240    # bottom-right

# Draw rectangle: image, (x1, y1), (x2, y2), color(B,G,R), thickness
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Changing Color and Thickness

```python
import cv2

img = cv2.imread("example.jpg")

# Blue rectangle, 4 px thick
cv2.rectangle(img, (30, 40), (180, 160), (255, 0, 0), 4)

# Red filled rectangle (thickness = -1 fills the shape)
cv2.rectangle(img, (200, 40), (350, 160), (0, 0, 255), -1)

cv2.imshow("Colors & Thickness", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Putting a Label on Top of the Box

```python
import cv2

img = cv2.imread("example.jpg")

# Box coordinates
x1, y1, x2, y2 = 60, 80, 320, 240

# Draw green rectangle
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Prepare label text
label = "Person: 0.92"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 2

# Get text size for background box
(text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
# Place label background above the top-left (adjust if it would go off-image)
label_x1, label_y1 = x1, max(0, y1 - text_h - baseline - 4)
label_x2, label_y2 = x1 + text_w + 6, y1

# Draw filled background (black)
cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)

# Put white text just above the box
cv2.putText(img, label, (x1 + 3, y1 - baseline - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

cv2.imshow("Rectangle with Label", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Convert `(x, y, w, h) to (x1, y1, x2, y2)`

```python
import cv2

img = cv2.imread("example.jpg")

# Suppose a detector returns (x, y, w, h)
x, y, w, h = 120, 90, 200, 140

# Convert to corners
x1, y1 = x, y
x2, y2 = x + w, y + h

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

cv2.imshow("(x, y, w, h) → corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Draw Rectangle on Video Frames from File

```python
import cv2

cap = cv2.VideoCapture("example.mp4")
if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Create a box centered in the frame (~40% width, 30% height)
    box_w, box_h = int(w * 0.4), int(h * 0.3)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Draw rectangle (cyan)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Label with background
    label = "Demo Box"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (tw, th), base = cv2.getTextSize(label, font, font_scale, thickness)
    bg_x1, bg_y1 = x1, max(0, y1 - th - base - 6)
    bg_x2, bg_y2 = x1 + tw + 8, y1

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 4, y1 - base - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("Video with Rectangle", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Drawing on a Webcam

```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 20, 20, w - 20, h - 20
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(frame, "Webcam Box", (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Webcam Rectangle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
