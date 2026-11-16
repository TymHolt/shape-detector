import cv2
import numpy as np

class ColorRange:
    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

class ObjectColor:
    def __init__(self, name, rgb, hsv_range, hsv_second_range=None):
        self.name = name
        self.rgb = rgb
        self.hsv_range = hsv_range
        self.hsv_second_range = hsv_second_range

OBJECT_COLORS = [
    ObjectColor("yellow", (0, 255, 255), ColorRange([15, 80, 80], [40, 255, 255])),
    ObjectColor("green", (0, 255, 0), ColorRange([40, 50, 50], [85, 255, 255])),
    ObjectColor("blue", (255, 0, 0), ColorRange([90, 50, 50], [130, 255, 255])),
    # Red has two ranges!
    ObjectColor("red", (0, 0, 255), ColorRange([0, 70, 50], [10, 255, 255]),
                ColorRange([170, 70, 50], [180, 255, 255]))
]