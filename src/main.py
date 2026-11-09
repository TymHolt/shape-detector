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

        key = cv2.waitKeyEx(10)

        if key == ord("a"):
            detection_pipeline.stage_selector -= 1
        elif key == ord("d"):
            detection_pipeline.stage_selector += 1
        elif key == ord("w"):
            detection_pipeline.section_selector -= 1
        elif key == ord("s"):
            detection_pipeline.section_selector += 1
        elif key == ord("q"):
            break
        elif key == ord("\t"):
            detection_pipeline.ui_render_flag = not detection_pipeline.ui_render_flag

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()