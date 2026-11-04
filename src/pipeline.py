import cv2
from background import *

class DetectionPipeline:
    def __init__(self, frame_size=500):
        self.frame_size = frame_size
        self.background_detection = BackgroundDetection()
        self.stage_selector = 0
        self.stage_buffer = []

    def handle_frame(self, frame):
        self.stage_buffer = []
        self.add_stage("Camera Input", frame)

        # Scale to pipeline frame size for performance
        resize_ratio = self.frame_size / frame.shape[0]
        frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
        self.add_stage("Camera Input Resized", frame_resized)

        # Mask the background
        background = self.background_detection.mask_background(self, frame_resized)

        # Other detections
        # ...

        # Draw detected contours
        self.render_contour(frame_resized, background.contour)

        # Get selected stage, clamp index for safety
        self.stage_selector = max(0, min(self.stage_selector, len(self.stage_buffer) - 1))
        title, stage_frame = self.stage_buffer[self.stage_selector]
        
        # Stage frame may have different color format, copy to
        output = stage_frame.copy()
        if len(output.shape) == 2 or output.shape[2] == 1:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        cv2.putText(output, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return output

    def move_selector(self, direction):
        self.stage_selector += direction

    def add_stage(self, title, frame):
        self.stage_buffer.append((title, frame));

    def render_contour(self, frame, contour):
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            for point in contour:
                cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)