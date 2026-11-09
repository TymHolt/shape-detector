import cv2
from pipeline import *

def main():
    capture_index = -1 # This way the capture will be updated
    capture_index_switch = 0
    capture = None
    detection_pipeline = DetectionPipeline()
    
    while True:
        if capture_index != capture_index_switch:
            if capture != None:
                capture.release()
            capture_index = capture_index_switch
            capture = cv2.VideoCapture(capture_index)

            if not capture.isOpened():
                print("Error: Camera cannot be accessed")
                exit()
    
            print("Camera accessed...")

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
        elif key == ord("0"):
            capture_index_switch = 0
        elif key == ord("1"):
            capture_index_switch = 1
        elif key == ord("2"):
            capture_index_switch = 2
        elif key == ord("3"):
            capture_index_switch = 3
        elif key == ord("4"):
            capture_index_switch = 4
        elif key == ord("5"):
            capture_index_switch = 5
        elif key == ord("6"):
            capture_index_switch = 6
        elif key == ord("7"):
            capture_index_switch = 7
        elif key == ord("8"):
            capture_index_switch = 8
        elif key == ord("9"):
            capture_index_switch = 9

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()