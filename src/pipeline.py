import cv2
from background import *
from objects import *

class PipelineSection:
    def __init__(self, title):
        self.title = title
        self.stages = []

    def add_stage(self, title, frame):
        self.stages.append((title, frame));

class DetectionPipeline:
    def __init__(self, frame_size=500):
        self.frame_size = frame_size
        self.background_detection = BackgroundDetection()
        self.object_detection = ObjectDetection()
        self.section_selector = 0
        self.stage_selector = 0
        self.ui_render_flag = False
        self.sections = []

    def add_section(self, title):
        section = PipelineSection(title)
        self.sections.append(section)
        return section

    def handle_frame(self, frame):
        self.sections = []
        main_section = self.add_section("Main")
        main_section.add_stage("Camera Input", frame)

        # Scale to pipeline frame size for performance
        resize_ratio = self.frame_size / frame.shape[0]
        frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
        main_section.add_stage("Camera Input Resized", frame_resized)

        # Mask the background
        background = self.background_detection.mask_background(self, frame_resized)

        # Object detections
        self.object_detection.detect(self, background.masked_frame)

        # Draw detected contours
        self.render_contour(frame_resized, background.contour)

        # Get selected stage, clamp index for safety
        self.section_selector = max(0, min(self.section_selector, len(self.sections) - 1))
        selected_section = self.sections[self.section_selector]
        self.stage_selector = max(0, min(self.stage_selector, len(selected_section.stages) - 1))
        stage_title, stage_frame = selected_section.stages[self.stage_selector]
        
        # Stage frame may have different color format, copy to
        output = stage_frame.copy()
        if len(output.shape) == 2 or output.shape[2] == 1:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        if self.ui_render_flag:
            self.render_ui(selected_section.title, stage_title, output)

        main_section.add_stage("Output", output)
        return output

    def render_contour(self, frame, contour):
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            for point in contour:
                cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)

    def render_ui(self, section_title, stage_title, output):
        cv2.putText(output, section_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, stage_title, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)