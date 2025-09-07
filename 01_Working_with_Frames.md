# Working with Frames

In computer vision, images are treated as frames...

Using `opencv-python` (imported as `cv2`), we can read images from a given file which gives us a numpy array of shape `(B, R, C)`.

## Reading a Single Frame

* `cv2.imread("path")` by default reads in color (BGR) format.
* Returns a NumPy array: `(height, width, channels)`.

```python
import cv2

# Read an image in color mode (default)
img = cv2.imread("example.jpg")

# Display image shape
print("Shape:", img.shape)
```

## Grayscale Reading

* `cv2.IMREAD_GRAYSCALE` loads the image with one channel only.
* Useful for preprocessing in computer vision tasks.

```python
import cv2

# Read an image in grayscale
img_gray = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)

print("Shape (Grayscale):", img_gray.shape)
```

## Displaying na Image

* `cv2.imshow("Window Name", image)` opens a window.
* `cv2.waitKey(0)` waits indefinitely until a key is pressed.
* Always call cv2.destroyAllWindows() to close windows.

```bash
import cv2

# Read an image in color mode (default)
img = cv2.imread("example.jpg")

# Show image in a window
cv2.imshow("Color Image", img)

# Wait for a key press and close
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Reading from Video File

* Metadata FPS: Uses `cv2.CAP_PROP_FPS` to read FPS stored in the video file header.
* **Runtime FPS:** Actively measures FPS during playback by counting frames over elapsed time.
* The runtime FPS is printed every second in the terminal.

```python
import cv2
import time

# Open video file
cap = cv2.VideoCapture("example.mp4")

# Get FPS from metadata (may be approximate)
fps_metadata = cap.get(cv2.CAP_PROP_FPS)
print("Metadata FPS:", fps_metadata)

# For runtime FPS calculation
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Calculate FPS every second
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time >= 1.0:
        fps_runtime = frame_count / elapsed_time
        print(f"Runtime FPS: {fps_runtime:.2f}")
        frame_count = 0
        prev_time = current_time

    # Display video frame
    cv2.imshow("Video Frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Working with Webcams

```python
import cv2

# Open the default webcam (0 = primary camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
