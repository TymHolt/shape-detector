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

        for name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            color_mask = cv2.inRange(hsv, lower, upper)

            # TODO Cleaner
            # 2 ranges for red
            if name == "red":
                mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
                color_mask = cv2.bitwise_or(color_mask, mask2)
                # Extra Closing for red
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            # Remove noise
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            section.add_stage(f"Mask {name}", color_mask)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

                if circularity > 0.82:
                    shape = "sphere"
                elif circularity > 0.65 and aspect < 1.6:
                    shape = "cylinder"
                elif sides == 4 and 0.8 < aspect < 1.4:
                    shape = "cube"
                elif sides >= 3 and sides <= 6 and aspect < 1.8:
                    shape = "pyramid"
                else:
                    shape = "unknown"

                objects.append({
                    "color": name,
                    "shape": shape,
                    "contour": c,
                    "area": area,
                    "circularity": circularity,
                    "sides": sides,
                    "aspect": aspect
                })

                # Text position
                x, y, _, _ = cv2.boundingRect(c)
                cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
                label = f"{name} {shape}"
                cv2.putText(result, label, (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        section.add_stage("Detected", result)
        return objects