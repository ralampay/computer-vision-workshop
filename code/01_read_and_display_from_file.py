import cv2

# Read image
img = cv2.imread("example.jpg")

if img is None:
    raise FileNotFoundError("Could not read example.jpg")

# Show image in a window
while True:
    cv2.imshow("Display Image", img)

    # Wait for key press (1 ms) and check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
