import cv2
import numpy as np
from src.yoloEval import BoxDesc


class PointMatcher:

    def __init__(self, ratio_thresh: float = 0.85):
        # fastThreshold=5 sprawi, że wykryje narożniki nawet na słabym kontraście małych aut
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

            # 1. NAJPIERW RYSOWANIE RAMEK - żeby zawsze były widoczne na dole!
            cv2.rectangle(frameL, (lx, ly), (lx + lw, ly + lh), pair_color, 2)
            cv2.rectangle(frameR, (rx, ry), (rx + rw, ry + rh), pair_color, 2)

            roiL = frameL[ly : ly + lh, lx : lx + lw]
            roiR = frameR[ry : ry + rh, rx : rx + rw]

            if roiL.size == 0 or roiR.size == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            if len(roiL.shape) == 3:
                roiL_gray = cv2.cvtColor(roiL, cv2.COLOR_BGR2GRAY)
                roiR_gray = cv2.cvtColor(roiR, cv2.COLOR_BGR2GRAY)
            else:
                roiL_gray, roiR_gray = roiL, roiR

            kpL, desL = self.orb.detectAndCompute(roiL_gray, None)
            kpR, desR = self.orb.detectAndCompute(roiR_gray, None)

            # 2. Zmieńmy warunek z '< 2' na '< 1' lub po prostu brak deskryptora
            if desL is None or desR is None or len(desL) == 0 or len(desR) == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            matches = self.bf.knnMatch(desL, desR, k=2)

            ptsL = []
            ptsR = []

            for match in matches:
                # Zabezpieczenie: knnMatch może zwrócić mniej niż 2 dopasowania dla małych obszarów
                if len(match) < 2:
                    continue

                m, n = match
                if m.distance < self.ratio_thresh * n.distance:
                    pt_l = kpL[m.queryIdx].pt
                    pt_r = kpR[m.trainIdx].pt

                    global_x_l = int(pt_l[0] + lx)
                    global_y_l = int(pt_l[1] + ly)

                    global_x_r = int(pt_r[0] + rx)
                    global_y_r = int(pt_r[1] + ry)

                    ptsL.append([global_x_l, global_y_l])
                    ptsR.append([global_x_r, global_y_r])

                    cv2.circle(frameL, (global_x_l, global_y_l), 3, pair_color, -1)
                    cv2.circle(frameR, (global_x_r, global_y_r), 3, pair_color, -1)

            return np.array(ptsL, dtype=np.float32), np.array(ptsR, dtype=np.float32)

    def findPoints(
        self,
        bb_pairs: list[tuple[BoxDesc, BoxDesc]],
        frameL: np.ndarray,
        frameR: np.ndarray,
        draw_lines: bool = True,
    ):
        """Główna metoda pętli - wywołuje __findPoint dla każdej pary i łączy obrazy."""
        debug_L = frameL.copy()
        debug_R = frameR.copy()

        all_ptsL = []
        all_ptsR = []
        pair_colors = []  # Lista do przechowywania kolorów dla przypisanych punktów

        for pairL, pairR in bb_pairs:
            # Generujemy JEDEN unikalny/losowy kolor dla CAŁEJ PARY (BB, punkty, linie)
            pair_color = tuple(
                map(int, np.random.randint(50, 255, size=3))
            )

            # Przekazujemy prawidlowe zmienne (pairL do pairL, pairR do pairR) i kolor
            ptsL, ptsR = self.__findPoint(
                pairL, pairR, debug_L, debug_R, pair_color
            )

            if len(ptsL) > 0:
                all_ptsL.append(ptsL)
                all_ptsR.append(ptsR)
                pair_colors.append(pair_color)

        # Łączymy lewy i prawy obraz side-by-side
        vis_frame = np.hstack((debug_L, debug_R))

        # Rysowanie linii łączących punkty z zachowaniem spójnego koloru pary
        if draw_lines and len(all_ptsL) > 0:
            width_offset = frameL.shape[1]

            for ptsL, ptsR, color in zip(all_ptsL, all_ptsR, pair_colors):
                for pL, pR in zip(ptsL, ptsR):
                    pt_l = (int(pL[0]), int(pL[1]))
                    pt_r = (int(pR[0]) + width_offset, int(pR[1]))

                    # Wykorzystujemy ten sam kolor dla linii łączących
                    cv2.line(vis_frame, pt_l, pt_r, color, 1, cv2.LINE_AA)

        return vis_frame, (all_ptsL, all_ptsR)