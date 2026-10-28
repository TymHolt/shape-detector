import cv2
from pipeline import *

def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Camera cannot be accessed")
        exit()

    print("Camera accessed...")

    detection_pipeline = DetectionPipeline()
    while True:
        success, frame = capture.read()
        if not success:
            print("Error: Frame could not be read")
            break

        cv2.imshow("Camera Feed", detection_pipeline.handle_frame(frame))

        if cv2.waitKey(10) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()