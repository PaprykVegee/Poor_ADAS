import cv2
import numpy as np
from src.yoloEval import BoxDesc


class BBSearcher:

    def __init__(self, threshold=0.1):
        self.threshold = threshold

        self.F = np.array([
            [1.25e-07, 4.12e-06, -3.14e-03],
            [-5.01e-06, 8.82e-08, 1.42e-02],
            [2.89e-03, -1.51e-02, 1.00e00],
        ])

        self.localSearchFactor = 0.5

    def __localExpSize(self, rightBB: BoxDesc) -> np.ndarray:
        x, y, w, h = rightBB.coord

        x_new = x - w * (self.localSearchFactor / 2.0)
        y_new = y - h * (self.localSearchFactor / 2.0)

        w_new = w * (1.0 + self.localSearchFactor)
        h_new = h * (1.0 + self.localSearchFactor)

        return np.array([x_new, y_new, w_new, h_new])

    def __isInBox(self, leftBB: BoxDesc, rightBB: BoxDesc) -> bool:
        if leftBB.cls != rightBB.cls:
            return False

        rx, ry, rw, rh = self.__localExpSize(rightBB)
        lx, ly, _, _ = leftBB.coord

        return (rx <= lx <= rx + rw) and (ry <= ly <= ry + rh)

    def __calculateShapeError(
        self, leftBB: BoxDesc, rightBB: BoxDesc
    ) -> float:
        _, _, lw, lh = leftBB.coord
        _, _, rw, rh = rightBB.coord

        if lh <= 0 or rh <= 0 or lw <= 0 or rw <= 0:
            return 1.0

        ARl = lw / lh
        ARr = rw / rh

        Sh = 1.0 - abs(lh - rh) / max(lh, rh)
        Sar = 1.0 - abs(ARl - ARr) / max(ARl, ARr)

        similarity = max(0.0, Sh * Sar)
        return 1.0 - similarity

    def __calculateEpipolarError(
        self, leftBB: BoxDesc, rightBB: BoxDesc
    ) -> float:
        xl = leftBB.coord[0] + leftBB.coord[2] / 2.0
        yl = leftBB.coord[1] + leftBB.coord[3] / 2.0

        xr = rightBB.coord[0] + rightBB.coord[2] / 2.0
        yr = rightBB.coord[1] + rightBB.coord[3] / 2.0

        pl = np.array([xl, yl, 1.0])
        lr = self.F @ pl

        a, b, c = lr[0], lr[1], lr[2]

        numerator = np.abs(a * xr + b * yr + c)
        denominator = np.sqrt(a**2 + b**2)

        if denominator == 0:
            return float("inf")

        return float(numerator / denominator)

    def __calculateVIoUError(self, leftBB: BoxDesc, rightBB: BoxDesc) -> float:
        y1_min = leftBB.coord[1]
        y1_max = leftBB.coord[1] + leftBB.coord[3]

        y2_min = rightBB.coord[1]
        y2_max = rightBB.coord[1] + rightBB.coord[3]

        intersection_min = max(y1_min, y2_min)
        intersection_max = min(y1_max, y2_max)

        intersection = max(0.0, intersection_max - intersection_min)

        if intersection == 0:
            return 1.0  

        union_min = min(y1_min, y2_min)
        union_max = max(y1_max, y2_max)

        union = union_max - union_min

        if union == 0:
            return 1.0

        vIoU = float(intersection / union)
        return 1.0 - vIoU  

    def __calculateMatchingError(
        self, leftBB: BoxDesc, rightBB: BoxDesc
    ) -> float:
        w_epi = 1.0
        w_shape = 5.0 
        w_viou = 15.0  

        e_epi = self.__calculateEpipolarError(leftBB, rightBB)
        e_shape = self.__calculateShapeError(leftBB, rightBB)
        e_viou = self.__calculateVIoUError(leftBB, rightBB)

        return w_epi * e_epi + w_shape * e_shape + w_viou * e_viou

    def pipeline(
        self, leftBBs: list[BoxDesc], rightBBs: list[BoxDesc]
    ) -> list[tuple]:
        matches = []

        for rightBB in rightBBs:
            best_error = float("inf")
            best_leftBB = None

            for leftBB in leftBBs:
                if not self.__isInBox(leftBB, rightBB):
                    print('debug')
                    continue

                current_error = self.__calculateMatchingError(leftBB, rightBB)

                if current_error < best_error:
                    best_error = current_error
                    best_leftBB = leftBB

            if best_leftBB is not None and best_error < self.threshold:
                matches.append((rightBB, best_leftBB))

        return matches