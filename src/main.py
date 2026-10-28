import cv2
from background import mask_background

def handle_frame(frame):
    # Scale to ~500px size for performance
    resize_ratio = 500 / frame.shape[0]
    frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
    
    cv2.imshow("Camera Feed", mask_background(frame_resized))

def main():
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

        handle_frame(frame)

        if cv2.waitKey(10) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()