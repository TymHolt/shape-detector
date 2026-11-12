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
        self.section_selector = 1000 # Default value, gets clamped to max for final view
        self.stage_selector = 1000  # Default value, gets clamped to max for final view
        self.ui_render_flag = False
        self.ui_view_sections = True
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
        selected_stage = selected_section.stages[self.stage_selector]
        
        # Stage frame may have different color format, copy to
        output = selected_stage[1].copy()
        if len(output.shape) == 2 or output.shape[2] == 1:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        if self.ui_render_flag:
            self.render_ui(selected_section, selected_stage, output)

        main_section.add_stage("Output", output)
        return output

    def render_contour(self, frame, contour):
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            for point in contour:
                cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)

    def ui_input(self, direction):
        if not self.ui_render_flag:
            return

        if self.ui_view_sections:
            self.section_selector += direction
            self.stage_selector = 1000 # Default value, gets clamped to max for final view
        else:
            self.stage_selector += direction

    def render_ui(self, current_section, current_stage, output):
        color_dark = (0, 150, 0)
        color_light = (100, 255, 100)

        if self.ui_view_sections:
            y = 30
            for section in self.sections:
                text = f"{section.title} >"
                text_color = color_dark
                if section.title == current_section.title:
                    text_color = color_light
                cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                y += 30
        else:
            cv2.putText(output, f"< {current_section.title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dark, 2, cv2.LINE_AA)           
            y = 60
            for stage in current_section.stages:
                text = f"{stage[0]}"
                text_color = color_dark
                if stage[0] == current_stage[0]:
                    text_color = color_light
                cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                y += 30