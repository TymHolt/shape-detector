import cv2
from background import *

class DetectionPipeline:
    def __init__(self, frame_size=500):
        self.frame_size = frame_size
        self.background_detection = BackgroundDetection()
        self.stage_selector = 0 # 0 is always the final result. 1, 2 ... are stages before
        self.stage_buffer = []

    def handle_frame(self, frame):
        self.stage_buffer = []
        self.add_stage("Camera Input", frame)

        # Scale to pipeline frame size for performance
        resize_ratio = self.frame_size / frame.shape[0]
        frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
        self.add_stage("Camera Input Resized", frame_resized)

        # Mask the background
        self.background_detection.mask_background(self, frame_resized)

        # Return selected stage, clamp index for safety
        self.stage_selector = max(0, min(self.stage_selector, len(self.stage_buffer) - 1))
        title, stage_frame = self.stage_buffer[self.stage_selector]
        cv2.putText(stage_frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return stage_frame

    def move_selector(self, direction):
        self.stage_selector += direction

    def add_stage(self, title, frame):
        self.stage_buffer.append((title, frame));