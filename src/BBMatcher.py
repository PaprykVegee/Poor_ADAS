import cv2
import numpy as np

from src.yoloEval import BoxDesc


class BBMatcher:
    """
    Klasa bazowa dla matcherów bboxów.

    Zawiera wspólne:
    - operacje na bboxach,
    - local search,
    - sprawdzanie klasy,
    - obliczanie odległości / różnicy rozmiaru.
    """

    def __init__(
        self,
        threshold: float = 500.0,
        local_search_x: float = 2.0,
        local_search_y: float = 1.0,
    ) -> None:
        self.threshold = threshold

        self.local_search_x = local_search_x
        self.local_search_y = local_search_y

    def _get_center(
        self,
        bb: BoxDesc,
    ) -> tuple[float, float]:

        x, y, w, h = bb.coord

        return (
            x + w / 2.0,
            y + h / 2.0,
        )

    def _get_bottom_center(
        self,
        bb: BoxDesc,
    ) -> tuple[float, float]:

        x, y, w, h = bb.coord

        return (
            x + w / 2.0,
            y + h,
        )

    def _get_area(
        self,
        bb: BoxDesc,
    ) -> float:

        _, _, w, h = bb.coord

        return w * h

    def _get_local_search_box(
        self,
        bb: BoxDesc,
    ) -> np.ndarray:
        """
        Zwraca lokalny obszar wyszukiwania wokół bboxa.

        local_search_x = 2 oznacza rozszerzenie:
            2 * szerokość bboxa w lewo
            2 * szerokość bboxa w prawo

        Analogicznie dla Y.
        """

        x, y, w, h = bb.coord

        x_new = x - w * self.local_search_x
        y_new = y - h * self.local_search_y

        w_new = w * (
            1.0 + 2.0 * self.local_search_x
        )

        h_new = h * (
            1.0 + 2.0 * self.local_search_y
        )

        return np.array(
            [
                x_new,
                y_new,
                w_new,
                h_new,
            ],
            dtype=np.float64,
        )

    def _is_in_local_search(
        self,
        reference_bb: BoxDesc,
        candidate_bb: BoxDesc,
    ) -> bool:
        """
        Sprawdza, czy candidate_bb znajduje się
        w lokalnym obszarze wyszukiwania reference_bb.
        """

        if reference_bb.cls != candidate_bb.cls:
            return False

        rx, ry, rw, rh = self._get_local_search_box(
            reference_bb
        )

        cx, cy = self._get_bottom_center(
            candidate_bb
        )

        return (
            rx <= cx <= rx + rw
            and
            ry <= cy <= ry + rh
        )

    def _calculate_position_error(
        self,
        bb0: BoxDesc,
        bb1: BoxDesc,
    ) -> float:
        """
        Odległość pomiędzy dolnymi środkami bboxów.
        """

        x0, y0 = self._get_bottom_center(bb0)
        x1, y1 = self._get_bottom_center(bb1)

        return float(
            np.hypot(
                x0 - x1,
                y0 - y1,
            )
        )

    def _calculate_size_error(
        self,
        bb0: BoxDesc,
        bb1: BoxDesc,
    ) -> float:
        """
        Różnica względna rozmiaru bboxów.
        """

        area0 = self._get_area(bb0)
        area1 = self._get_area(bb1)

        if area0 <= 0 or area1 <= 0:
            return 1.0

        return float(
            abs(area0 - area1)
            / max(area0, area1)
        )


class BBLeftRightMatcher(BBMatcher):
    """
    Matcher pomiędzy lewą i prawą kamerą.

    LEFT ↔ RIGHT
    """

    def __init__(
        self,
        threshold: float = 500.0,
        local_search_x: float = 1.0,
        local_search_y: float = 0.5,
        corr_weight: float = 300.0,
    ) -> None:

        super().__init__(
            threshold=threshold,
            local_search_x=local_search_x,
            local_search_y=local_search_y,
        )

        self.corr_weight = corr_weight

    def _calculate_correlation_error(
        self,
        bb0: BoxDesc,
        bb1: BoxDesc,
        img0: np.ndarray,
        img1: np.ndarray,
    ) -> float:

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        x0, y0, bw0, bh0 = [
            int(v) for v in bb0.coord
        ]

        x1, y1, bw1, bh1 = [
            int(v) for v in bb1.coord
        ]

        crop0 = img0[
            max(0, y0):min(h0, y0 + bh0),
            max(0, x0):min(w0, x0 + bw0),
        ]

        crop1 = img1[
            max(0, y1):min(h1, y1 + bh1),
            max(0, x1):min(w1, x1 + bw1),
        ]

        if crop0.size == 0 or crop1.size == 0:
            return 1.0

        if len(crop0.shape) == 3:
            crop0 = cv2.cvtColor(
                crop0,
                cv2.COLOR_BGR2GRAY,
            )

        if len(crop1.shape) == 3:
            crop1 = cv2.cvtColor(
                crop1,
                cv2.COLOR_BGR2GRAY,
            )

        target_size = (64, 64)

        crop0 = cv2.resize(
            crop0,
            target_size,
        )

        crop1 = cv2.resize(
            crop1,
            target_size,
        )

        crop0 = self._pre_process(crop0)
        crop1 = self._pre_process(crop1)

        F0 = np.fft.fft2(crop0)
        F1 = np.fft.fft2(crop1)

        numerator = F0 * np.conj(F1)

        denominator = (
            np.abs(numerator) + 1e-5
        )

        response_complex = np.fft.ifft2(
            numerator / denominator
        )

        response = np.abs(
            response_complex
        )

        max_response = np.max(response)

        corr_error = 1.0 - max_response

        return max(
            0.0,
            float(corr_error),
        )

    def _pre_process(
        self,
        img: np.ndarray,
    ) -> np.ndarray:

        height, width = img.shape

        img = img.astype(
            np.float32
        )

        img = np.log(
            img + 1.0
        )

        mean = np.mean(img)
        std = np.std(img)

        if std != 0:
            img = (
                img - mean
            ) / std

        win_col = np.hanning(width)
        win_row = np.hanning(height)

        mask_col, mask_row = np.meshgrid(
            win_col,
            win_row,
        )

        window = (
            mask_col * mask_row
        )

        return img * window

    def _calculate_matching_error(
        self,
        left_bb: BoxDesc,
        right_bb: BoxDesc,
        img_left: np.ndarray,
        img_right: np.ndarray,
    ) -> float:

        _, y_left = self._get_bottom_center(
            left_bb
        )

        _, y_right = self._get_bottom_center(
            right_bb
        )

        e_y = abs(
            y_left - y_right
        )

        e_size = self._calculate_size_error(
            left_bb,
            right_bb,
        )

        x_left, _ = self._get_bottom_center(
            left_bb
        )

        x_right, _ = self._get_bottom_center(
            right_bb
        )

        disparity = x_left - x_right

        e_corr = self._calculate_correlation_error(
            left_bb,
            right_bb,
            img_left,
            img_right,
        )

        return (
            e_y * 5.0
            + e_size * 100.0
            + abs(disparity) * 0.05
            + e_corr * self.corr_weight
        )

    def pipeline(
        self,
        left_bbs: list[BoxDesc],
        right_bbs: list[BoxDesc],
        img_left: np.ndarray,
        img_right: np.ndarray,
    ) -> list[tuple[BoxDesc, BoxDesc]]:

        matches = []

        for right_bb in right_bbs:

            best_error = float("inf")
            best_left_bb = None

            for left_bb in left_bbs:

                if not self._is_in_local_search(
                    right_bb,
                    left_bb,
                ):
                    continue

                current_error = (
                    self._calculate_matching_error(
                        left_bb,
                        right_bb,
                        img_left,
                        img_right,
                    )
                )

                if current_error < best_error:
                    best_error = current_error
                    best_left_bb = left_bb

            if (
                best_left_bb is not None
                and best_error < self.threshold
            ):
                matches.append(
                    (
                        best_left_bb,
                        right_bb,
                    )
                )

        return matches



class FrameMatcher(BBMatcher):
    """
    Matcher pomiędzy kolejnymi klatkami.

    frame t <-> frame t+1

    Utrzymuje track_id dla obiektów i pamięta tracki,
    które chwilowo nie zostały wykryte przez YOLO.

    max_age:
        Maksymalna liczba kolejnych klatek, przez które
        track może nie mieć detekcji, zanim zostanie usunięty.
    """

    def __init__(
        self,
        threshold: float = 300.0,
        local_search_x: float = 3.0,
        local_search_y: float = 2.0,
        position_weight: float = 1.0,
        size_weight: float = 100.0,
        max_age: int = 5,
    ) -> None:

        super().__init__(
            threshold=threshold,
            local_search_x=local_search_x,
            local_search_y=local_search_y,
        )

        self.position_weight = position_weight
        self.size_weight = size_weight

        self.max_age = max_age

        self.tracks: dict[int, dict] = {}

        self.next_track_id = 0

    def _calculate_frame_matching_error(
        self,
        previous_bb: BoxDesc,
        current_bb: BoxDesc,
    ) -> float:

        if previous_bb.cls != current_bb.cls:
            return float("inf")

        position_error = (
            self._calculate_position_error(
                previous_bb,
                current_bb,
            )
        )

        size_error = (
            self._calculate_size_error(
                previous_bb,
                current_bb,
            )
        )

        return (
            position_error * self.position_weight
            +
            size_error * self.size_weight
        )

    def _create_track(
        self,
        bb: BoxDesc,
    ) -> int:

        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "bbox": bb,
            "misses": 0,
        }

        return track_id

    def _remove_old_tracks(self) -> None:
        """
        Usuwa tracki, które były niewykryte przez max_age klatek.
        """

        tracks_to_remove = []

        for track_id, track in self.tracks.items():

            if track["misses"] > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def pipeline(
        self,
        previous_bbs: list[BoxDesc],
        current_bbs: list[BoxDesc],
        frame_0: np.ndarray | None = None,
        frame_1: np.ndarray | None = None,
    ) -> list[
        tuple[int, BoxDesc | None, BoxDesc | None]
    ]:

        matches = []

        if len(self.tracks) == 0:

            for current_bb in current_bbs:

                track_id = self._create_track(
                    current_bb
                )

                matches.append(
                    (
                        track_id,
                        None,
                        current_bb,
                    )
                )

            return matches

        active_tracks = list(
            self.tracks.items()
        )

        used_tracks = set()

        for current_bb in current_bbs:

            best_error = float("inf")
            best_track_id = None

            for track_id, track in active_tracks:

                if track_id in used_tracks:
                    continue

                previous_bb = track["bbox"]

                if previous_bb.cls != current_bb.cls:
                    continue

                if not self._is_in_local_search(
                    previous_bb,
                    current_bb,
                ):
                    continue


                current_error = (
                    self._calculate_frame_matching_error(
                        previous_bb,
                        current_bb,
                    )
                )

                if current_error < best_error:

                    best_error = current_error
                    best_track_id = track_id


            if (
                best_track_id is not None
                and best_error < self.threshold
            ):

                track = self.tracks[
                    best_track_id
                ]

                previous_bb = track["bbox"]

                track["bbox"] = current_bb

                track["misses"] = 0

                used_tracks.add(
                    best_track_id
                )

                matches.append(
                    (
                        best_track_id,
                        previous_bb,
                        current_bb,
                    )
                )


            else:

                track_id = self._create_track(
                    current_bb
                )

                used_tracks.add(
                    track_id
                )

                matches.append(
                    (
                        track_id,
                        None,
                        current_bb,
                    )
                )


        for track_id, track in active_tracks:

            if track_id in used_tracks:
                continue

            track["misses"] += 1
            matches.append(
                (
                    track_id,
                    track["bbox"],
                    None,
                )
            )

        self._remove_old_tracks()

        return matches