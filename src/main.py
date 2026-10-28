import cv2

def handle_frame(frame):
    # Edge detection works best with grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurr, smooth out details and noise
    frame_blurred = cv2.GaussianBlur(frame_grayscale, (5, 5), 0)

    # Filter edges with given threshold
    frame_edges = cv2.Canny(frame_blurred, 75, 200)

    cv2.imshow("Camera Feed", frame_edges)


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Camera cannot be accessed")
    exit()

print("Camera accessed...")

while True:
    success, frame = capture.read()
    if not success:
        print("Error: Frame could not be read")
        break

    # Scale to ~500px size for performance
    resize_ratio = 500 / frame.shape[0]
    frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)

    handle_frame(frame_resized)

    if cv2.waitKey(10) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
