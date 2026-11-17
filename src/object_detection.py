import cv2
import numpy as np
from object_classes import *

class ObjectDetection:
    def __init__(self, object_min_area=150):
        self.object_min_area = object_min_area

    def detect(self, pipeline, masked_frame):
        section = pipeline.gui.add_section("Objects")

        # Turn frame into HSV
        hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        section.add_stage("HSV", hsv_frame)

        objects = []
        result = masked_frame.copy()

        # Object detection for each color
        for object_color in OBJECT_COLORS:
            mask_stage = section.add_stage(f"Mask {object_color.name}", hsv_frame)
            sat_lower_adjuster = mask_stage.add_adjuster(f"Saturation (Lower) {object_color.name}: ", 0, 50, 255, 5)
            sat_upper_adjuster = mask_stage.add_adjuster(f"Saturation (Upper) {object_color.name}: ", 0, 255, 255, 5)
            v_lower_adjuster = mask_stage.add_adjuster(f"Light (Lower) {object_color.name}: ", 0, 50, 255, 5)
            v_upper_adjuster = mask_stage.add_adjuster(f"Light (Upper) {object_color.name}: ", 0, 255, 255, 5)
                                    
            color_mask_frame = self.create_color_mask(hsv_frame, object_color, sat_lower_adjuster.current, sat_upper_adjuster.current,
                                                      v_lower_adjuster.current, v_upper_adjuster.current)
            mask_stage.frame = color_mask_frame

            # Find contours
            contours, _ = cv2.findContours(color_mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_tree, hierarchy = cv2.findContours(color_mask_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get object attributes from contour
                object_attributes = self.get_object_attributes(contour)
                if object_attributes == None:
                    continue

                # Identify object shape
                shape = "unknown"

                # Spherical Object
                if object_attributes.sides >= 7 and object_attributes.circularity >= 0.70:
                    if object_attributes.aspect >= 1.1:
                        shape = "cylinder"                            
                    elif object_attributes.circularity < 0.75:
                        shape = "cylinder"        
                    else:
                        shape = "sphere"
                else: # Non-Spehrical object
                    if object_attributes.sides <= 5 and object_attributes.aspect < 1.1:
                        shape = "pyramid"
                    else:
                        shape = "cube"

                shaped_object = ShapedObject(object_color, object_attributes, shape, contour)
                objects.append(shaped_object)
                position = self.get_normalized_position(pipeline.background.contour, contour)

                # Label
                label = shaped_object.as_string()

                if position != None:
                    label = f"{label} ({position[0]:.2f}, {position[1]:.2f})"

                label_color = (255, 255, 255)
                if pipeline.gui.visible:
                    label = object_attributes.as_string()
                else:
                    label_color = object_color.rgb

                cv2.drawContours(result, [contour], -1, label_color, 2)

                for pt in object_attributes.approx:
                    cv2.circle(result, tuple(pt[0]), 5, (255, 0, 255), -1)

                if hierarchy is not None:
                    for i, cnt in enumerate(contours_tree):
                        if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > 30:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                                    cv2.drawContours(result, [cnt], -1, (255, 100, 255), 1)

                bx, by, _, _ = cv2.boundingRect(contour)
                cv2.putText(result, label, (bx, max(by - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, label_color, 1)

        section.add_stage("Detected", result)
        return objects

    def get_object_attributes(self, contour):
        area = cv2.contourArea(contour)
        if area < self.object_min_area:
            return None

        object_attributes = ObjectAttributes()

        # Sides
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        object_attributes.approx = approx
        object_attributes.sides = len(approx)

        # Circularity
        object_attributes.circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        # Aspect
        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        object_attributes.aspect = max(rw, rh) / (min(rw, rh) + 1e-5)

        return object_attributes

    def get_normalized_position(self, background_contour, object_contour):
        px, py, pw, ph = cv2.boundingRect(background_contour)
        
        M = cv2.moments(object_contour)
        if M["m00"] == 0:
            return None
        ox = M["m10"] / M["m00"]
        oy = M["m01"] / M["m00"]
        
        nx = (ox - px) / pw
        ny = (oy - py) / ph
        
        nx = np.clip(nx, 0.0, 1.0)
        ny = np.clip(ny, 0.0, 1.0)
        
        return (nx, ny)

    def create_color_mask(self, hsv_frame, object_color, sat_lower, sat_upper, v_lower, v_upper):
        hsv_range = object_color.hsv_range
        hsv_scond_range = object_color.hsv_second_range
        color_mask = cv2.inRange(hsv_frame, hsv_range.lower, hsv_range.upper)

        if hsv_scond_range != None:
            mask2 = cv2.inRange(hsv_frame, hsv_scond_range.lower, hsv_scond_range.upper)
            color_mask = cv2.bitwise_or(color_mask, mask2)

        saturation_mask = cv2.inRange(hsv_frame[:, :, 1], sat_lower, sat_upper)
        color_mask = cv2.bitwise_and(color_mask, saturation_mask)

        v_mask = cv2.inRange(hsv_frame[:, :, 2], v_lower, v_upper)
        color_mask = cv2.bitwise_and(color_mask, v_mask)

        kernel_open = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

        kernel_close = np.ones((7, 7), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        return color_mask