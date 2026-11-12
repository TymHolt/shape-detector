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

            if color_name == "red":
                mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
                color_mask = cv2.bitwise_or(color_mask, mask2)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            sat_mask = cv2.inRange(hsv[:, :, 1], 35, 255)
            color_mask = cv2.bitwise_and(color_mask, sat_mask)

            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            section.add_stage(f"Mask {color_name}", color_mask)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_tree, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.025 * peri, True)
                sides = len(approx)

                circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

                rect = cv2.minAreaRect(c)
                (rw, rh) = rect[1]
                aspect = max(rw, rh) / (min(rw, rh) + 1e-5)

                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                sides_pts = approx.reshape(-1, 2)
                if len(sides_pts) >= 3:
                    side_lengths = [np.linalg.norm(sides_pts[i] - sides_pts[(i + 1) % len(sides_pts)]) for i in range(len(sides_pts))]
                    side_var = np.var(side_lengths) / (np.mean(side_lengths) + 1e-5)
                else:
                    side_var = 0

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
                    angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                    angles.append(angle)

                min_angle = min(angles) if angles else 180
                angle_var = np.var(angles) if angles else 0

                inner_count = 0
                if hierarchy is not None:
                    for i, cnt in enumerate(contours_tree):
                        if hierarchy[0][i][3] != -1:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                if cv2.pointPolygonTest(c, (cx, cy), False) >= 0 and cv2.contourArea(cnt) > 40:
                                    inner_count += 1

                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                dists = np.linalg.norm(c.reshape(-1, 2) - np.array([cx, cy]), axis=1)
                mean_dist = np.mean(dists)
                max_dist = np.max(dists)
                pointiness = max_dist / (mean_dist + 1e-5)          # > 1.4–1.6 oft Pyramid
                dist_var = np.var(dists) / (mean_dist + 1e-5)

                if sides >= 7 and circularity >= 0.80 and solidity > 0.92 and aspect < 1.28:
                    shape = "sphere"
                elif sides >= 6 and circularity >= 0.70 and circularity < 0.80 and solidity > 0.87 and pointiness < 1.55:
                    shape = "cylinder"
                elif sides == 4 and side_var < 0.25 and pointiness < 1.45 and 0.60 < circularity < 0.76 and solidity > 0.88:
                    shape = "pyramid"
                elif pointiness >= 1.45 or (side_var > 0.40 and sides <= 6):
                    shape = "cube"
                else:
                    shape = "unknown"

                objects.append({
                    "color": color_name,
                    "shape": shape,
                    "contour": c,
                    "area": area,
                    "sides": sides,
                    "circularity": circularity,
                    "side_var": side_var,
                    "solidity": solidity,
                    "min_angle": min_angle,
                    "angle_var": angle_var,
                    "inner_count": inner_count,
                    "aspect": aspect
                })

                # Label
                label = f"{color_name} {shape}"
                label_color = (255, 255, 255)
                if pipeline.ui_render_flag:
                    label = f"{color_name} {shape} s={sides} c={circularity:.2f} var={side_var:.2f} point={pointiness:.2f}"
                else:
                    if color_name == "red":
                        label_color = (0, 0, 255)
                    elif color_name == "green":
                        label_color = (0, 255, 0)
                    elif color_name == "blue":
                        label_color = (255, 0, 0)
                    elif color_name == "yellow":
                        label_color = (0, 255, 255)

                cv2.drawContours(result, [c], -1, label_color, 2)

                for pt in approx:
                    cv2.circle(result, tuple(pt[0]), 5, (255, 0, 255), -1)

                if hierarchy is not None:
                    for i, cnt in enumerate(contours_tree):
                        if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > 30:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                if cv2.pointPolygonTest(c, (cx, cy), False) >= 0:
                                    cv2.drawContours(result, [cnt], -1, (255, 100, 255), 1)

                bx, by, _, _ = cv2.boundingRect(c)
                cv2.putText(result, label, (bx, max(by - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, label_color, 1)

        section.add_stage("Detected", result)
        return objects