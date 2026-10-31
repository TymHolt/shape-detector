import cv2
from background import *

class DetectionPipeline:
    def __init__(self, frame_size=500):
        self.frame_size = frame_size
        self.background_detection = BackgroundDetection()

    def handle_frame(self, frame):
        # Scale to pipeline frame size for performance
        resize_ratio = self.frame_size / frame.shape[0]
        frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)

        return self.background_detection.mask_background(frame_resized).mask