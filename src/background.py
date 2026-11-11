import cv2
import numpy as np

class Background:
    def __init__(self, mask, masked_frame, contour):
        self.mask = mask
        self.masked_frame = masked_frame
        self.contour = contour

class BackgroundDetection:
    def __init__(self, blurr_size = 5, canny_low = 50, canny_high = 150):
        self.blurr_size = blurr_size
        self.canny_low = canny_low
        self.canny_high = canny_high

    def mask_background(self, pipeline, frame):
        section = pipeline.add_section("Background")

        # Edge detection works best with grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        section.add_stage("Grayscale", frame_grayscale)

        # Blurr, smooth out details and noise
        frame_blurred = cv2.GaussianBlur(frame_grayscale, (self.blurr_size, self.blurr_size), 0)
        section.add_stage("Blurred", frame_blurred)

        # Filter edges with given threshold
        frame_edges = cv2.Canny(frame_blurred, self.canny_low, self.canny_high)
        section.add_stage("Canny", frame_edges)

        # Close edge gaps and corrections
        kernel = np.ones((5, 5), np.uint8)
        frame_edges = cv2.dilate(frame_edges, kernel, iterations=2)
        section.add_stage("Dilate", frame_edges)
        frame_edges = cv2.erode(frame_edges, kernel, iterations=1)
        section.add_stage("Erode", frame_edges)

        # Find contours
        contours, _ = cv2.findContours(frame_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        background_contour = None
        background_area = -1
        for contour in contours:
            # Search biggest area
            area = cv2.contourArea(contour)
            if area < background_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)        
            if len(approx) == 4:
                background_contour = approx
                background_area = area
                break

        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if background_contour is not None:
            cv2.drawContours(mask, [background_contour], -1, 255, -1)
        section.add_stage("Mask", mask)

        # Clean edges of mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        section.add_stage("Masked", masked_frame)

        return Background(mask, masked_frame, background_contour)