import numpy as np
import cv2


import cv2
import numpy as np


class TriangulationPkt:

    def __init__(
        self,
        eps: float = 0.0,
        K: np.ndarray = np.array(
            [[881.0, 0.0, 440.5], [0.0, 881.0, 200.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    ) -> None:

        self.eps = eps

        R1 = np.eye(3)
        T1 = np.zeros((3, 1))

        R2 = np.eye(3)
        T2 = np.array([[-0.5], [0.0], [0.0]])

        self.P1 = K @ np.hstack((R1, T1))
        self.P2 = K @ np.hstack((R2, T2))

    def __triangulation(
        self, ptL: tuple[float, float], ptR: tuple[float, float]
    ) -> np.ndarray:
        pt_l = np.array(ptL, dtype=np.float64).reshape(2, 1)
        pt_r = np.array(ptR, dtype=np.float64).reshape(2, 1)

        point_4d = cv2.triangulatePoints(self.P1, self.P2, pt_l, pt_r)
        point_3d = point_4d[:3] / point_4d[3]

        return point_3d  # Zwraca [[X], [Y], [Z]]

    def __filter_range(
        self, data: np.ndarray, percentage: float = 0.15
    ) -> np.ndarray:
        if len(data) == 0:
            return data

        mean_val = np.mean(data)

        lower_bound = mean_val * (1.0 - percentage)
        upper_bound = mean_val * (1.0 + percentage)

        mask = (data >= lower_bound) & (data <= upper_bound)

        return data[mask]

    def pipeline(
        self, ptsL: list[tuple[int, int]], ptsR: list[tuple[int, int]]
    ) -> float:
        z_distances = []

        for ptL, ptR in zip(ptsL, ptsR):
            point_3d = self.__triangulation(ptL, ptR)
            z_depth = point_3d[2][0]
            z_distances.append(z_depth)

        z_array = np.array(z_distances)

        filtered_z = self.__filter_range(z_array, percentage=0.15)

        if len(filtered_z) == 0:
            return 0.0

        return float(np.mean(filtered_z))

