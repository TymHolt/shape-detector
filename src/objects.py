import cv2
import numpy as np

class ObjectDetection:
    def __init__(self):
        self.color_ranges = {
            "yellow": ([15, 80, 80], [40, 255, 255]),
            "green": ([40, 50, 50], [85, 255, 255]),
            "blue": ([90, 50, 50], [130, 255, 255]),
            "red": ([0, 70, 50], [10, 255, 255]),
        }
        self.red_lower2 = np.array([170, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])
        self.min_area = 150

    def detect(self, pipeline, masked_frame):
        section = pipeline.add_section("Objects")

        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        section.add_stage("HSV", hsv)

        objects = []
        result = masked_frame.copy()

        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            color_mask = cv2.inRange(hsv, lower, upper)

            # 2 ranges for red
            if color_name == "red":
                mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
                color_mask = cv2.bitwise_or(color_mask, mask2)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            # Saturation filter against reflections
            sat_mask = cv2.inRange(hsv[:, :, 1], 40, 255)
            color_mask = cv2.bitwise_and(color_mask, sat_mask)

            # Remove noise
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            section.add_stage(f"Mask {color_name}", color_mask)

            # External contours for objects
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Inner contours (for visualization)
            contours_tree, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue

                # Shape analysis
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                sides = len(approx)

                circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

                rect = cv2.minAreaRect(c)
                (rw, rh) = rect[1]
                aspect = max(rw, rh) / (min(rw, rh) + 1e-5)

                # Additional features
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                x, y, w, h = cv2.boundingRect(c)
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0

                # Side length variance
                sides_pts = approx.reshape(-1, 2)
                side_lengths = []
                for i in range(len(sides_pts)):
                    p1 = sides_pts[i]
                    p2 = sides_pts[(i + 1) % len(sides_pts)]
                    side_lengths.append(np.linalg.norm(p1 - p2))
                side_var = np.var(side_lengths) / (np.mean(side_lengths) + 1e-5) if side_lengths else 0

                # Convexity defects
                hull_idx = cv2.convexHull(c, returnPoints=False)
                defect_count = 0
                if hull_idx is not None and len(hull_idx) > 3 and len(c) > 3:
                    defects = cv2.convexityDefects(c, hull_idx)
                    defect_count = 0 if defects is None else len(defects)

                # Angles between sides
                angles = []
                pts = approx.reshape(-1, 2)
                for i in range(len(pts)):
                    p0 = pts[i - 1]
                    p1 = pts[i]
                    p2 = pts[(i + 1) % len(pts)]
                    v1 = p0 - p1
                    v2 = p2 - p1
                    norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5
                    cosang = np.dot(v1, v2) / norm
                    angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
                angle_var = np.var(angles) if angles else 0

                # Classification
                if sides == 4 and 0.85 < aspect < 1.35 and solidity > 0.90 and side_var < 0.40 and defect_count < 3:
                    shape = "cube"
                elif sides >= 6 and circularity > 0.65 and aspect < 1.5 and defect_count <= 2:
                    shape = "cylinder"
                elif side_var > 0.50 or defect_count >= 3 or (sides == 4 and side_var > 0.30):
                    shape = "pyramid"
                elif circularity > 0.85 and 0.85 < aspect < 1.15:
                    shape = "sphere"
                else:
                    shape = "unknown"

                objects.append({
                    "color": color_name,
                    "shape": shape,
                    "contour": c,
                    "area": area,
                    "circularity": circularity,
                    "sides": sides,
                    "aspect": aspect,
                    "defect_count": defect_count,
                    "angle_var": angle_var
                })

                # Label color logic
                label = f"{color_name} {shape}"
                label_color = (255, 255, 255)
                if pipeline.ui_render_flag:
                    label = f"{color_name} {shape} s={sides} def={defect_count} ang={angle_var:.0f} var={side_var:.2f}"
                else:
                    if color_name == "red":
                        label_color = (0, 0, 255)
                    elif color_name == "green":
                        label_color = (0, 255, 0)
                    elif color_name == "blue":
                        label_color = (255, 0, 0)
                    elif color_name == "yellow":
                        label_color = (0, 255, 255)
                    else:
                        label_color = (255, 255, 255)

                # Draw main contour
                cv2.drawContours(result, [c], -1, label_color, 2)

                # Highlight corners (approx points)
                for point in approx:
                    cv2.circle(result, tuple(point[0]), 4, (0, 255, 255), -1)

                # Highlight inner contours (thinner, different color)
                if hierarchy is not None:
                    for i, cnt in enumerate(contours_tree):
                        if hierarchy[0][i][3] != -1:  # has parent → inner
                            if cv2.contourArea(cnt) > 50:
                                cv2.drawContours(result, [cnt], -1, (255, 0, 255), 1)

                # Text
                x, y, _, _ = cv2.boundingRect(c)
                cv2.putText(result, label, (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

        section.add_stage("Detected", result)
        return objects