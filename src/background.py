import cv2

def mask_background(frame):
    # Edge detection works best with grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurr, smooth out details and noise
    frame_blurred = cv2.GaussianBlur(frame_grayscale, (5, 5), 0)

    # Filter edges with given threshold
    frame_edges = cv2.Canny(frame_blurred, 75, 200)

    return frame_edges