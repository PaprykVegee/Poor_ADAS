
import cv2
import numpy as np
from src.characteristicPointMatcher import ChPtrDesc


class TriangulationPkt:
    def __init__(
        self,
        K: np.ndarray = np.array(
            [
                [881.0, 0.0, 440.5],
                [0.0, 881.0, 200.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        baseline: float = 0.5,
        max_reprojection_error: float = 2.0,
        min_distance: float = 0.5,
        max_distance: float = 200.0,
    ) -> None:

        self.K = K.astype(np.float64)

        self.baseline = baseline
        self.max_reprojection_error = max_reprojection_error
        self.min_distance = min_distance
        self.max_distance = max_distance

        R1 = np.eye(3, dtype=np.float64)
        T1 = np.zeros((3, 1), dtype=np.float64)

        R2 = np.eye(3, dtype=np.float64)
        T2 = np.array(
            [[baseline], [0.0], [0.0]],
            dtype=np.float64,
        )

        self.P1 = self.K @ np.hstack((R1, T1))
        self.P2 = self.K @ np.hstack((R2, T2))

        self.R1 = R1
        self.T1 = T1
        self.R2 = R2
        self.T2 = T2

    def __triangulate_point(
        self,
        ptL: np.ndarray,
        ptR: np.ndarray,
    ) -> np.ndarray | None:
        """
        Trianguluje pojedynczą parę punktów.

        Zwraca:
            XYZ w układzie lewej kamery
        """

        pt_l = np.asarray(ptL, dtype=np.float64).reshape(2, 1)
        pt_r = np.asarray(ptR, dtype=np.float64).reshape(2, 1)

        point_4d = cv2.triangulatePoints(
            self.P1,
            self.P2,
            pt_l,
            pt_r,
        )

        w = point_4d[3, 0]

        if abs(w) < 1e-6:
            return None

        point_3d = point_4d[:3, 0] / w

        if not np.all(np.isfinite(point_3d)):
            return None

        return point_3d

    def __project_point(
        self,
        point_3d: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray | None:
        """
        Projekcja punktu 3D z powrotem na obraz.
        """

        point_4d = np.append(point_3d, 1.0)

        projected = P @ point_4d

        w = projected[2]

        if abs(w) < 1e-6:
            return None

        projected_2d = projected[:2] / w

        if not np.all(np.isfinite(projected_2d)):
            return None

        return projected_2d

    def __reprojection_error(
        self,
        point_3d: np.ndarray,
        ptL: np.ndarray,
        ptR: np.ndarray,
    ) -> tuple[float, float] | None:
        """
        Oblicza błąd reprojekcji dla lewej i prawej kamery.

        Zwraca:
            (error_left, error_right)
        """

        projected_L = self.__project_point(
            point_3d,
            self.P1,
        )

        projected_R = self.__project_point(
            point_3d,
            self.P2,
        )

        if projected_L is None or projected_R is None:
            return None

        ptL = np.asarray(ptL, dtype=np.float64).reshape(2)
        ptR = np.asarray(ptR, dtype=np.float64).reshape(2)

        error_L = float(
            np.linalg.norm(ptL - projected_L)
        )

        error_R = float(
            np.linalg.norm(ptR - projected_R)
        )

        return error_L, error_R

    def __is_valid_3d_point(
        self,
        point_3d: np.ndarray,
    ) -> bool:
        """
        Podstawowy sanity check punktu 3D.
        """

        if not np.all(np.isfinite(point_3d)):
            return False

        z_left = point_3d[2]

        if not (
            self.min_distance
            < z_left
            < self.max_distance
        ):
            return False

        point_right = point_3d - self.T2.reshape(3)

        z_right = point_right[2]

        if z_right <= 0:
            return False

        return True

    def __filter_iqr(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        """
        Filtracja wartości odstających metodą IQR.
        """

        if len(distances) < 4:
            return distances

        q25 = np.percentile(distances, 25)
        q75 = np.percentile(distances, 75)

        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        valid_mask = (
            (distances >= lower_bound)
            & (distances <= upper_bound)
        )

        return distances[valid_mask]

    def process_descriptor(
        self,
        desc: ChPtrDesc,
    ) -> ChPtrDesc:

        if (
            desc.pointsL is None
            or desc.pointsR is None
            or len(desc.pointsL) == 0
            or len(desc.pointsR) == 0
        ):
            desc.triangulation_value = None
            return desc

        if len(desc.pointsL) != len(desc.pointsR):
            desc.triangulation_value = None
            return desc

        z_distances = []

        for ptL, ptR in zip(
            desc.pointsL,
            desc.pointsR,
        ):

            point_3d = self.__triangulate_point(
                ptL,
                ptR,
            )

            if point_3d is None:
                continue

            if not self.__is_valid_3d_point(
                point_3d
            ):
                continue

            reprojection = self.__reprojection_error(
                point_3d,
                ptL,
                ptR,
            )

            if reprojection is None:
                continue

            error_L, error_R = reprojection

            if (
                error_L > self.max_reprojection_error
                or error_R > self.max_reprojection_error
            ):
                continue


            z = float(point_3d[2])

            z_distances.append(z)

        if len(z_distances) == 0:
            desc.triangulation_value = None
            return desc

        z_arr = np.asarray(
            z_distances,
            dtype=np.float64,
        )


        filtered_z = self.__filter_iqr(
            z_arr
        )

        if len(filtered_z) == 0:
            desc.triangulation_value = None
            return desc

        desc.triangulation_value = float(
            np.median(filtered_z)
        )

        return desc
