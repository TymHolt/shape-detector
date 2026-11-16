import cv2
import numpy as np
from object_classes import *

class ObjectDetection:
    def __init__(self, object_min_area=150):
        self.object_min_area = object_min_area

    def detect(self, pipeline, masked_frame):
        section = pipeline.add_section("Objects")

        # Turn frame into HSV
        hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        section.add_stage("HSV", hsv_frame)

        objects = []
        result = masked_frame.copy()

        # Object detection for each color
        for object_color in OBJECT_COLORS:
            color_mask_frame = self.create_color_mask(hsv_frame, object_color)
            section.add_stage(f"Mask {object_color.name}", color_mask_frame)

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
                if object_attributes.sides >= 7 and object_attributes.circularity >= 0.80 and object_attributes.solidity > 0.92 and object_attributes.aspect < 1.28:
                    shape = "sphere"
                elif object_attributes.sides >= 6 and object_attributes.circularity >= 0.70 and object_attributes.circularity < 0.80 and object_attributes.solidity > 0.87 and object_attributes.pointiness < 1.55:
                    shape = "cylinder"
                elif object_attributes.sides == 4 and object_attributes.side_var < 0.25 and object_attributes.pointiness < 1.45 and 0.60 < object_attributes.circularity < 0.76 and object_attributes.solidity > 0.88:
                    shape = "pyramid"
                elif object_attributes.pointiness >= 1.45 or (object_attributes.side_var > 0.40 and object_attributes.sides <= 6):
                    shape = "cube"    

                shaped_object = ShapedObject(object_color, object_attributes, shape, contour)
                objects.append(shaped_object)

                # Label
                label = shaped_object.as_string()
                label_color = (255, 255, 255)
                if pipeline.ui_render_flag:
                    label = object_attributes.as_string()
                else:
                    label_color = object_color.rgb

                cv2.drawContours(result, [contour], -1, label_color, 2)

                # TODO
                # for pt in approx:
                #    cv2.circle(result, tuple(pt[0]), 5, (255, 0, 255), -1)

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
        object_attributes.sides = len(approx)

        # Circularity
        object_attributes.circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        # Aspect
        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        object_attributes.aspect = max(rw, rh) / (min(rw, rh) + 1e-5)

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        object_attributes.solidity = area / hull_area if hull_area > 0 else 0

        # Side var
        sides_pts = approx.reshape(-1, 2)
        if len(sides_pts) >= 3:
            side_lengths = [np.linalg.norm(sides_pts[i] - sides_pts[(i + 1) % len(sides_pts)]) for i in range(len(sides_pts))]
            object_attributes.side_var = np.var(side_lengths) / (np.mean(side_lengths) + 1e-5)
        else:
            object_attributes.side_var = 0

        # Pointiness
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dists = np.linalg.norm(contour.reshape(-1, 2) - np.array([cx, cy]), axis=1)
        mean_dist = np.mean(dists)
        max_dist = np.max(dists)
        object_attributes.pointiness = max_dist / (mean_dist + 1e-5)

        return object_attributes

    def create_color_mask(self, hsv_frame, object_color):
        hsv_range = object_color.hsv_range
        hsv_scond_range = object_color.hsv_second_range
        color_mask = cv2.inRange(hsv_frame, hsv_range.lower, hsv_range.upper)

        if hsv_scond_range != None:
            mask2 = cv2.inRange(hsv_frame, hsv_scond_range.lower, hsv_scond_range.upper)
            color_mask = cv2.bitwise_or(color_mask, mask2)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        sat_mask = cv2.inRange(hsv_frame[:, :, 1], 35, 255)
        color_mask = cv2.bitwise_and(color_mask, sat_mask)

        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        return color_mask