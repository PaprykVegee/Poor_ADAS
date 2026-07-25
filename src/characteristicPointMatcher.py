import cv2
import numpy as np
from src.yoloEval import BoxDesc


class PointMatcher:

    def __init__(self, ratio_thresh: float = 1):
        self.orb = cv2.ORB_create(nfeatures=1000, fastThreshold=1, scoreType=cv2.ORB_FAST_SCORE)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_thresh = ratio_thresh

    def __findPoint(
            self,
            pairL: BoxDesc,
            pairR: BoxDesc,
            frameL: np.ndarray,
            frameR: np.ndarray,
            pair_color: tuple[int, int, int],
        ):
            lx, ly, lw, lh = map(int, pairL.coord)
            rx, ry, rw, rh = map(int, pairR.coord)

            roiL = frameL[ly : ly + lh, lx : lx + lw]
            roiR = frameR[ry : ry + rh, rx : rx + rw]

            cv2.rectangle(frameL, (lx, ly), (lx + lw, ly + lh), pair_color, 2)
            cv2.rectangle(frameR, (rx, ry), (rx + rw, ry + rh), pair_color, 2)

            if roiL.size == 0 or roiR.size == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            if len(roiL.shape) == 3:
                roiL_gray = cv2.cvtColor(roiL, cv2.COLOR_RGB2GRAY)
                roiR_gray = cv2.cvtColor(roiR, cv2.COLOR_RGB2GRAY)
            else:
                roiL_gray, roiR_gray = roiL, roiR

            min_dim_L = min(roiL_gray.shape[:2])
            min_dim_R = min(roiR_gray.shape[:2])

            scale_L = 1.0
            scale_R = 1.0

            if min_dim_L < 80:
                scale_L = 80.0 / min_dim_L
                roiL_gray = cv2.resize(roiL_gray, (0, 0), fx=scale_L, fy=scale_L, interpolation=cv2.INTER_CUBIC)

            if min_dim_R < 80:
                scale_R = 80.0 / min_dim_R
                roiR_gray = cv2.resize(roiR_gray, (0, 0), fx=scale_R, fy=scale_R, interpolation=cv2.INTER_CUBIC)

            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            roiL_gray = cv2.filter2D(roiL_gray, -1, kernel)
            roiR_gray = cv2.filter2D(roiR_gray, -1, kernel)

            kpL, desL = self.orb.detectAndCompute(roiL_gray, None)
            kpR, desR = self.orb.detectAndCompute(roiR_gray, None)

            if desL is None or desR is None or len(desL) == 0 or len(desR) == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            matches = self.bf.knnMatch(desL, desR, k=2)

            ptsL = []
            ptsR = []

            for match in matches:
                if len(match) < 2:
                    if len(match) == 1:
                        m = match[0]
                        pt_l = kpL[m.queryIdx].pt
                        pt_r = kpR[m.trainIdx].pt
                    else:
                        continue
                else:
                    m, n = match
                    if m.distance >= self.ratio_thresh * n.distance:
                        continue
                    pt_l = kpL[m.queryIdx].pt
                    pt_r = kpR[m.trainIdx].pt

                orig_pt_l_x = pt_l[0] / scale_L
                orig_pt_l_y = pt_l[1] / scale_L

                orig_pt_r_x = pt_r[0] / scale_R
                orig_pt_r_y = pt_r[1] / scale_R

                global_x_l = int(orig_pt_l_x + lx)
                global_y_l = int(orig_pt_l_y + ly)

                global_x_r = int(orig_pt_r_x + rx)
                global_y_r = int(orig_pt_r_y + ry)

                ptsL.append([global_x_l, global_y_l])
                ptsR.append([global_x_r, global_y_r])

                cv2.circle(frameL, (global_x_l, global_y_l), 2, pair_color, -1)
                cv2.circle(frameR, (global_x_r, global_y_r), 2, pair_color, -1)

            return np.array(ptsL, dtype=np.float32), np.array(ptsR, dtype=np.float32)

    def findPoints(
        self,
        bb_pairs: list[tuple[BoxDesc, BoxDesc]],
        frameL: np.ndarray,
        frameR: np.ndarray,
        draw_lines: bool = True,
    ):
        debug_L = frameL.copy()
        debug_R = frameR.copy()

        all_ptsL = []
        all_ptsR = []
        pair_colors = [] 

        for pairL, pairR in bb_pairs:
            pair_color = tuple(
                map(int, np.random.randint(50, 255, size=3))
            )

            ptsL, ptsR = self.__findPoint(
                pairL, pairR, debug_L, debug_R, pair_color
            )

            if len(ptsL) > 0:
                all_ptsL.append(ptsL)
                all_ptsR.append(ptsR)
                pair_colors.append(pair_color)

        vis_frame = np.hstack((debug_L, debug_R))

        if draw_lines and len(all_ptsL) > 0:
            width_offset = frameL.shape[1]

            for ptsL, ptsR, color in zip(all_ptsL, all_ptsR, pair_colors):
                for pL, pR in zip(ptsL, ptsR):
                    pt_l = (int(pL[0]), int(pL[1]))
                    pt_r = (int(pR[0]) + width_offset, int(pR[1]))

                    cv2.line(vis_frame, pt_l, pt_r, color, 1, cv2.LINE_AA)

        return vis_frame, (all_ptsL, all_ptsR)