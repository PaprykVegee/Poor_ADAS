import cv2
import numpy as np
from src.characteristicPointMatcher import ChPtrDesc

class TriangulationPkt:
    def __init__(
        self,
        K: np.ndarray = np.array(
            [[881.0, 0.0, 440.5], [0.0, 881.0, 200.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        baseline: float = 0.5,
    ) -> None:
        self.fx = K[0, 0]  # Ogniskowa w pikselach
        self.baseline = baseline
        
        R1, T1 = np.eye(3), np.zeros((3, 1))
        R2, T2 = np.eye(3), np.array([[baseline], [0.0], [0.0]])

        self.P1 = K @ np.hstack((R1, T1))
        self.P2 = K @ np.hstack((R2, T2))

    def __triangulate_point(self, ptL: np.ndarray, ptR: np.ndarray) -> float:
        pt_l = np.array(ptL, dtype=np.float64).reshape(2, 1)
        pt_r = np.array(ptR, dtype=np.float64).reshape(2, 1)

        point_4d = cv2.triangulatePoints(self.P1, self.P2, pt_l, pt_r)
        w = point_4d[3][0]
        if abs(w) < 1e-6:
            return -1.0
        return point_4d[2][0] / w

    def __estimate_from_bb(self, coordL: list, coordR: list) -> float:
        """Fallback: Obliczanie odległości na podstawie przesunięcia ramek YOLO."""
        if coordL is None or coordR is None:
            return None
        
        lx, ly, lw, lh = coordL
        rx, ry, rw, rh = coordR

        # Różnica współrzędnej X środków obu ramek
        center_x_l = lx + (lw / 2.0)
        center_x_r = rx + (rw / 2.0)
        disparity = center_x_l - center_x_r

        if disparity <= 0.1:  # Brak poprawnej dysparycji
            return None

        # Wzór podstawowy stereo: Z = (f * B) / disparity
        z = (self.fx * self.baseline) / disparity
        return z

    def process_descriptor(self, desc: ChPtrDesc) -> ChPtrDesc:
        z_distances = []

        # 1. Główna ścieżka: Triangulacja z punktów ORB/RANSAC
        if desc.pointsL is not None and desc.pointsR is not None and len(desc.pointsL) > 0:
            for ptL, ptR in zip(desc.pointsL, desc.pointsR):
                z = self.__triangulate_point(ptL, ptR)
                if 0.5 < z < 200.0:
                    z_distances.append(z)

        # 2. Jeśli mamy punkty po triangulacji, używamy IQR + Mediany
        if len(z_distances) >= 3:
            q25, q75 = np.percentile(z_distances, [25, 75])
            iqr = q75 - q25
            valid_mask = (z_distances >= q25 - 1.5 * iqr) & (z_distances <= q75 + 1.5 * iqr)
            valid_z = np.array(z_distances)[valid_mask]
            
            if len(valid_z) > 0:
                desc.triangulation_value = float(np.median(valid_z))
                return desc

        # 3. FALLBACK: Brak punktów charakterystycznych -> Szacujemy z Bounding Boxa!
        bb_z = self.__estimate_from_bb(desc.coordL, desc.coordR)
        if bb_z is not None and 0.5 < bb_z < 200.0:
            desc.triangulation_value = float(bb_z)
        else:
            desc.triangulation_value = None

        return desc