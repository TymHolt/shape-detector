import cv2
from pipeline_classes import *
from background import *
from object_detection import *

class DetectionPipeline:
    def __init__(self, frame_size=500):
        self.frame_size = frame_size
        self.background_detection = BackgroundDetection()
        self.object_detection = ObjectDetection()
        self.gui = PipelineGui()

    def handle_frame(self, frame):
        main_section = self.gui.add_section("Main")
        main_section.add_stage("Camera Input", frame)

        # Scale to pipeline frame size for performance
        resize_ratio = self.frame_size / frame.shape[0]
        frame_resized = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
        main_section.add_stage("Camera Input Resized", frame_resized)

        # Mask the background
        background = self.background_detection.mask_background(self, frame_resized)

        # Object detections
        self.object_detection.detect(self, background.masked_frame)

        self.gui.clamp_all_indices()
        selected_section = self.gui.get_current_section()
        selected_stage = self.gui.get_current_stage()

        # Stage frame may have different color format, copy to
        output = selected_stage.frame.copy()
        if len(output.shape) == 2 or output.shape[2] == 1:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        if self.gui.visible:
            self.render_ui(selected_section, selected_stage, output)
            
        return output

    def render_ui(self, current_section, current_stage, output):
        color_dark = (0, 150, 0)
        color_light = (100, 255, 100)

        if self.gui.view_depth == VIEW_DEPTH_SECTION:
            y = 30
            for section in self.gui.sections:
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
                text = f"{stage.title}"
                text_color = color_dark
                if stage.title == current_stage.title:
                    text_color = color_light
                cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                y += 30