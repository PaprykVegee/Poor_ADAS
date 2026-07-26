import cv2
import numpy as np
from src.yoloEval import BoxDesc


class ChPtrDesc:
    def __init__(
        self,
        coordL: list[int | float] = None,
        coordR: list[int | float] = None,
        cls: str = None,
        pointsL: np.ndarray = None,
        pointsR: np.ndarray = None,
        triangulation_value: float = None,
    ):
        self.coordL = coordL
        self.coordR = coordR
        self.cls = cls
        self.pointsL = pointsL
        self.pointsR = pointsR
        self.triangulation_value = triangulation_value


class PointMatcher:
    def __init__(self, ratio_thresh: float = 0.8, ransac_thresh: float = 1.0):
        self.orb = cv2.ORB_create(nfeatures=1000, fastThreshold=1, scoreType=cv2.ORB_FAST_SCORE)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh  # Próg RANSAC (tolerancja błędu w pikselach)

    def __findPoint(
        self, pairL: BoxDesc, pairR: BoxDesc, frameL: np.ndarray, frameR: np.ndarray, pair_color: tuple
    ):
        lx, ly, lw, lh = map(int, pairL.coord)
        rx, ry, rw, rh = map(int, pairR.coord)

        roiL = frameL[ly : ly + lh, lx : lx + lw]
        roiR = frameR[ry : ry + rh, rx : rx + rw]

        cv2.rectangle(frameL, (lx, ly), (lx + lw, ly + lh), pair_color, 2)
        cv2.rectangle(frameR, (rx, ry), (rx + rw, ry + rh), pair_color, 2)

        if roiL.size == 0 or roiR.size == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        roiL_gray = cv2.cvtColor(roiL, cv2.COLOR_RGB2GRAY) if len(roiL.shape) == 3 else roiL
        roiR_gray = cv2.cvtColor(roiR, cv2.COLOR_RGB2GRAY) if len(roiR.shape) == 3 else roiR

        scale_L = 80.0 / min(roiL_gray.shape[:2]) if min(roiL_gray.shape[:2]) < 80 else 1.0
        scale_R = 80.0 / min(roiR_gray.shape[:2]) if min(roiR_gray.shape[:2]) < 80 else 1.0

        if scale_L != 1.0:
            roiL_gray = cv2.resize(roiL_gray, (0, 0), fx=scale_L, fy=scale_L, interpolation=cv2.INTER_CUBIC)
        if scale_R != 1.0:
            roiR_gray = cv2.resize(roiR_gray, (0, 0), fx=scale_R, fy=scale_R, interpolation=cv2.INTER_CUBIC)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        roiL_gray = cv2.filter2D(roiL_gray, -1, kernel)
        roiR_gray = cv2.filter2D(roiR_gray, -1, kernel)

        kpL, desL = self.orb.detectAndCompute(roiL_gray, None)
        kpR, desR = self.orb.detectAndCompute(roiR_gray, None)

        if desL is None or desR is None or len(desL) == 0 or len(desR) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        matches = self.bf.knnMatch(desL, desR, k=2)

        raw_ptsL, raw_ptsR = [], []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance >= self.ratio_thresh * n.distance:
                    continue
            elif len(match) == 1:
                m = match[0]
            else:
                continue

            pt_l = kpL[m.queryIdx].pt
            pt_r = kpR[m.trainIdx].pt

            global_x_l = int((pt_l[0] / scale_L) + lx)
            global_y_l = int((pt_l[1] / scale_L) + ly)
            global_x_r = int((pt_r[0] / scale_R) + rx)
            global_y_r = int((pt_r[1] / scale_R) + ry)

            raw_ptsL.append([global_x_l, global_y_l])
            raw_ptsR.append([global_x_r, global_y_r])

        ptsL_arr = np.array(raw_ptsL, dtype=np.float32)
        ptsR_arr = np.array(raw_ptsR, dtype=np.float32)

        if len(ptsL_arr) >= 8:
            _, mask = cv2.findFundamentalMat(
                ptsL_arr, ptsR_arr, cv2.FM_RANSAC, self.ransac_thresh, 0.99
            )
            
            if mask is not None:
                inliers_mask = mask.ravel() == 1
                ptsL_arr = ptsL_arr[inliers_mask]
                ptsR_arr = ptsR_arr[inliers_mask]

        for pL, pR in zip(ptsL_arr, ptsR_arr):
            cv2.circle(frameL, (int(pL[0]), int(pL[1])), 2, pair_color, -1)
            cv2.circle(frameR, (int(pR[0]), int(pR[1])), 2, pair_color, -1)

        return ptsL_arr, ptsR_arr

    def findPoints(
        self,
        bb_pairs: list[tuple[BoxDesc, BoxDesc]],
        frameL: np.ndarray,
        frameR: np.ndarray,
        draw_lines: bool = True,
    ) -> tuple[np.ndarray, list[ChPtrDesc]]:
        debug_L = frameL.copy()
        debug_R = frameR.copy()
        descriptors: list[ChPtrDesc] = []

        for pairL, pairR in bb_pairs:
            pair_color = tuple(map(int, np.random.randint(50, 255, size=3)))
            ptsL, ptsR = self.__findPoint(pairL, pairR, debug_L, debug_R, pair_color)

            desc = ChPtrDesc(
                coordL=pairL.coord,
                coordR=pairR.coord,
                cls=pairL.cls,
                pointsL=ptsL,
                pointsR=ptsR,
                triangulation_value=None,
            )
            descriptors.append(desc)

        vis_frame = np.hstack((debug_L, debug_R))

        if draw_lines:
            width_offset = frameL.shape[1]
            for desc in descriptors:
                if len(desc.pointsL) > 0:
                    for pL, pR in zip(desc.pointsL, desc.pointsR):
                        pt_l = (int(pL[0]), int(pL[1]))
                        pt_r = (int(pR[0]) + width_offset, int(pR[1]))
                        cv2.line(vis_frame, pt_l, pt_r, (0, 255, 0), 1, cv2.LINE_AA)

        return vis_frame, descriptors