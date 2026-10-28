import cv2

class BackgroundDetection:
    def __init__(self, blurr_size = 5, canny_low = 75, canny_high = 200):
        self.blurr_size = blurr_size
        self.canny_low = canny_low
        self.canny_high = canny_high

    def mask_background(self, frame):
        # Edge detection works best with grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blurr, smooth out details and noise
        frame_blurred = cv2.GaussianBlur(frame_grayscale, (self.blurr_size, self.blurr_size), 0)

        # Filter edges with given threshold
        frame_edges = cv2.Canny(frame_blurred, self.canny_low, self.canny_high)

        return frame_edges