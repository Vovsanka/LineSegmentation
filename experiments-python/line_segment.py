import math

import numpy as np


class LineSegment:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def direction(self) -> np.ndarray:
        return np.array([self.x2 - self.x1, self.y2 - self.y1])

    def point_to_line_dist(self, px, py):
        A = px - self.x1
        B = py - self.y1
        C = self.x2 - self.x1
        D = self.y2 - self.y1
        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq < 1e-9:
            return math.hypot(px - self.x1, py - self.y1)

        param = dot / len_sq
        if param < 0:
            xx, yy = self.x1, self.y1
        elif param > 1:
            xx, yy = self.x2, self.y2
        else:
            xx = self.x1 + param * C
            yy = self.y1 + param * D

        return math.hypot(px - xx, py - yy)
        