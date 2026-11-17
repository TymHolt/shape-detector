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
    ObjectColor("yellow", (0, 255, 255), ColorRange([15, 0, 80], [40, 255, 255])),
    ObjectColor("green", (0, 255, 0), ColorRange([40, 0, 50], [85, 255, 255])),
    ObjectColor("blue", (255, 0, 0), ColorRange([90, 0, 50], [130, 255, 255])),
    # Red has two ranges!
    ObjectColor("red", (0, 0, 255), ColorRange([0, 0, 50], [10, 255, 255]),
                ColorRange([170, 0, 50], [180, 255, 255]))
]

class ObjectAttributes:
    def __init__(self, sides=0, circularity=0, aspect=0, approx=None):
        self.sides = sides
        self.circularity = circularity
        self.aspect = aspect
        self.approx = approx

    def as_string(self):
        return f"sides={self.sides} circ={self.circularity:.2f} aspect={self.aspect:.2f}"

class ShapedObject:
    def __init__(self, color, attributes, shape, contour):
        self.color = color
        self.attributes = attributes
        self.shape = shape
        self.contour = contour

    def as_string(self):
        return f"{self.color.name} {self.shape}"