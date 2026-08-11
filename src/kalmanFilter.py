
import cv2
import numpy as np


class DepthKalmanFilter:
    """
    Filtr Kalmana dla pojedynczego track_id.

    Stan:
        x = [distance, velocity]

    distance -> odległość obiektu w metrach
    velocity -> zmiana odległości [m/s]
    """

    def __init__(
        self,
        dt: float = 0.4,
        process_noise: float = 0.05,
        measurement_noise: float = 0.5,
    ) -> None:

        self.kalman = cv2.KalmanFilter(
            2,
            1,
            0,
            cv2.CV_64F,
        )

        # State transition:
        #
        # distance_new = distance + velocity * dt
        # velocity_new = velocity
        self.kalman.transitionMatrix = np.array(
            [
                [1.0, dt],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Measurement:
        #
        # mierzymy tylko distance
        self.kalman.measurementMatrix = np.array(
            [
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.kalman.processNoiseCov = (
            np.eye(2, dtype=np.float64)
            * process_noise
        )

        self.kalman.measurementNoiseCov = np.array(
            [
                [measurement_noise],
            ],
            dtype=np.float64,
        )

        self.kalman.errorCovPost = (
            np.eye(2, dtype=np.float64)
        )

        self.initialized = False

    def update(
        self,
        measurement: float | None,
    ) -> float | None:

        # Pierwszy poprawny pomiar inicjalizuje filtr.
        if not self.initialized:

            if measurement is None:
                return None

            self.kalman.statePost = np.array(
                [
                    [measurement],
                    [0.0],
                ],
                dtype=np.float64,
            )

            self.initialized = True

            return float(measurement)

        # Najpierw predykcja.
        prediction = self.kalman.predict()

        # Jeżeli nie mamy pomiaru,
        # zwracamy wartość przewidzianą przez filtr.
        if measurement is None:

            return float(
                prediction[0, 0]
            )

        # Aktualizacja pomiarem.
        measurement_array = np.array(
            [
                [measurement],
            ],
            dtype=np.float64,
        )

        corrected = self.kalman.correct(
            measurement_array
        )

        return float(
            corrected[0, 0]
        )


class KalmanManager:
    """
    Zarządza osobnym filtrem Kalmana dla każdego track_id.
    """

    def __init__(
        self,
        dt: float = 0.4,
        process_noise: float = 0.05,
        measurement_noise: float = 0.5,
    ) -> None:

        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.filters: dict[int, DepthKalmanFilter] = {}

    def update(
        self,
        track_id: int,
        measurement: float | None,
    ) -> float | None:

        if track_id not in self.filters:

            self.filters[track_id] = (
                DepthKalmanFilter(
                    dt=self.dt,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            )

        return self.filters[track_id].update(
            measurement
        )

    def remove(
        self,
        track_id: int,
    ) -> None:

        self.filters.pop(
            track_id,
            None,
        )

    def clear(self) -> None:

        self.filters.clear()
