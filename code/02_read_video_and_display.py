import cv2

# Open video file
cap = cv2.VideoCapture("example-video.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Could not open example-video.mp4")

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        # End of video
        break

    # Display the frame
    cv2.imshow("Video Playback", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
