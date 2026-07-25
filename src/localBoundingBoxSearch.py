import cv2
import numpy as np
from src.yoloEval import BoxDesc


class BBSearcher:
    def __init__(
        self,
        threshold: float = 500.0,
        max_y_diff: float = 50.0,
        max_size_ratio: float = 2.0,
        corr_weight: float = 300.0,  
    ):
        self.threshold = threshold
        self.max_y_diff = max_y_diff
        self.max_size_ratio = max_size_ratio
        self.corr_weight = corr_weight

    def __get_bottom_center(self, bb: BoxDesc) -> tuple[float, float]:
        x, y, w, h = bb.coord
        cx = x + w / 2.0
        cy = y + h
        return cx, cy

    def __localExpSize(self, rightBB: BoxDesc) -> np.ndarray:
        x, y, w, h = rightBB.coord

        x_factor_left = 1.0
        x_factor_right = 3.5
        y_factor = 1.0

        x_new = x - w * x_factor_left
        y_new = y - h * (y_factor / 2.0)

        w_new = w * (1.0 + x_factor_left + x_factor_right)
        h_new = h * (1.0 + y_factor)

        return np.array([x_new, y_new, w_new, h_new])

    def __isInBox(self, leftBB: BoxDesc, rightBB: BoxDesc) -> bool:
        if leftBB.cls != rightBB.cls:
            return False

        rx, ry, rw, rh = self.__localExpSize(rightBB)
        lx, ly, lw, lh = leftBB.coord
        lmx = lx + lw / 2.0
        lmy = ly + lh / 2.0

        return (rx <= lmx <= rx + rw) and (ry <= lmy <= ry + rh)

    def __pre_process(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape
        img = img.astype(np.float32)

        img = np.log(img + 1.0)

        mean = np.mean(img)
        std = np.std(img)
        if std != 0:
            img = (img - mean) / std

        win_col = np.hanning(width)
        win_row = np.hanning(height)
        mask_col, mask_row = np.meshgrid(win_col, win_row)
        window = mask_col * mask_row

        return img * window

    def __calculateCorrelationError(
        self,
        leftBB: BoxDesc,
        rightBB: BoxDesc,
        img_left: np.ndarray,
        img_right: np.ndarray,
    ) -> float:
        h_l, w_l = img_left.shape[:2]
        h_r, w_r = img_right.shape[:2]

        lx, ly, lw, lh = [int(v) for v in leftBB.coord]
        rx, ry, rw, rh = [int(v) for v in rightBB.coord]

        crop_l = img_left[
            max(0, ly) : min(h_l, ly + lh), max(0, lx) : min(w_l, lx + lw)
        ]
        crop_r = img_right[
            max(0, ry) : min(h_r, ry + rh), max(0, rx) : min(w_r, rx + rw)
        ]

        if crop_l.size == 0 or crop_r.size == 0:
            return 1.0  

        if len(crop_l.shape) == 3:
            crop_l = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
        if len(crop_r.shape) == 3:
            crop_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)

        target_size = (64, 64)
        crop_l = cv2.resize(crop_l, target_size)
        crop_r = cv2.resize(crop_r, target_size)

        fi_l = self.__pre_process(crop_l)
        fi_r = self.__pre_process(crop_r)

        F_l = np.fft.fft2(fi_l)
        F_r = np.fft.fft2(fi_r)

        num = F_l * np.conj(F_r)
        den = np.abs(num) + 1e-5
        response_complex = np.fft.ifft2(num / den)
        response = np.abs(response_complex)

        max_response = np.max(response)

        corr_error = 1.0 - max_response
        return max(0.0, float(corr_error))

    def __calculateMatchingError(
        self,
        leftBB: BoxDesc,
        rightBB: BoxDesc,
        img_left: np.ndarray,
        img_right: np.ndarray,
    ) -> float:
        cxl, cyl = self.__get_bottom_center(leftBB)
        cxr, cyr = self.__get_bottom_center(rightBB)

        e_y = abs(cyl - cyr)

        _, _, lw, lh = leftBB.coord
        _, _, rw, rh = rightBB.coord
        area_l = lw * lh
        area_r = rw * rh
        e_size = abs(area_l - area_r) / max(area_l, area_r)

        disparity = cxl - cxr

        e_corr = self.__calculateCorrelationError(
            leftBB, rightBB, img_left, img_right
        )

        return (
            (e_y * 5.0)
            + (e_size * 100.0)
            + (abs(disparity) * 0.05)
            + (e_corr * self.corr_weight)
        )

    def pipeline(
        self,
        leftBBs: list[BoxDesc],
        rightBBs: list[BoxDesc],
        img_left: np.ndarray,
        img_right: np.ndarray,
    ) -> list[tuple[BoxDesc, BoxDesc]]:
        matches = []

        for rightBB in rightBBs:
            best_error = float("inf")
            best_leftBB = None

            for leftBB in leftBBs:
                if not self.__isInBox(leftBB, rightBB):
                    continue

                current_error = self.__calculateMatchingError(
                    leftBB, rightBB, img_left, img_right
                )

                if current_error < best_error:
                    best_error = current_error
                    best_leftBB = leftBB

            if best_leftBB is not None and best_error < self.threshold:
                matches.append((best_leftBB, rightBB))

        return matches