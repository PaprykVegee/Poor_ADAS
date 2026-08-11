import cv2
import numpy as np

from src.characteristicPointMatcher import PointMatcher
from src.triangulationCharacteristicsPoint import TriangulationPkt
from src.yoloEval import Inference
from src.BBMatcher import BBLeftRightMatcher, FrameMatcher


class ObjectCentricStereo:

    def __init__(
        self,
        model_path: str = "models_weights/yolo11n.pt",

        threshold: float = 500.0,
        corr_weight: float = 300.0,
        lowa_ratio: float = 0.9,
        baseline: float = 0.5,


        max_age: int = 5,


        kalman_dt: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 1.0,

    ) -> None:

        self.inference = Inference(
            model_path
        )


        self.bbSearcher = BBLeftRightMatcher(
            threshold=threshold,
            corr_weight=corr_weight,
        )


        self.ptrMatcher = PointMatcher(
            ratio_thresh=lowa_ratio
        )


        self.tranPkt = TriangulationPkt(
            baseline=baseline
        )

        self.frameMatcher = FrameMatcher(
            max_age=max_age
        )

        self.kalmans: dict[
            int,
            cv2.KalmanFilter
        ] = {}

        self.kalman_dt = kalman_dt

        self.kalman_process_noise = (
            kalman_process_noise
        )

        self.kalman_measurement_noise = (
            kalman_measurement_noise
        )

    def _create_kalman(
        self,
        initial_z: float,
    ) -> cv2.KalmanFilter:

        kalman = cv2.KalmanFilter(
            2,
            1,
            0,
            cv2.CV_64F,
        )

        kalman.transitionMatrix = np.array(
            [
                [
                    1.0,
                    self.kalman_dt,
                ],
                [
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        kalman.measurementMatrix = np.array(
            [
                [
                    1.0,
                    0.0,
                ],
            ],
            dtype=np.float64,
        )

        kalman.processNoiseCov = (
            np.eye(
                2,
                dtype=np.float64,
            )
            * self.kalman_process_noise
        )

        kalman.measurementNoiseCov = np.array(
            [
                [
                    self.kalman_measurement_noise
                ],
            ],
            dtype=np.float64,
        )

        kalman.errorCovPost = (
            np.eye(
                2,
                dtype=np.float64,
            )
        )

        kalman.statePost = np.array(
            [
                [
                    initial_z
                ],
                [
                    0.0
                ],
            ],
            dtype=np.float64,
        )

        return kalman

    def _update_kalman(
        self,
        track_id: int,
        measured_z: float | None,
    ) -> float | None:

        if track_id not in self.kalmans:

            if measured_z is None:
                return None

            self.kalmans[track_id] = (
                self._create_kalman(
                    measured_z
                )
            )

            return measured_z

        kalman = self.kalmans[track_id]

        prediction = kalman.predict()

        predicted_z = float(
            prediction[0, 0]
        )

        if measured_z is None:

            return predicted_z

        measurement = np.array(
            [
                [
                    measured_z
                ],
            ],
            dtype=np.float64,
        )

        corrected = kalman.correct(
            measurement
        )

        filtered_z = float(
            corrected[0, 0]
        )

        return filtered_z

    def _remove_kalman(
        self,
        track_id: int,
    ) -> None:

        self.kalmans.pop(
            track_id,
            None
        )


    @staticmethod
    def _bbox_key(bb):

        x, y, w, h = bb.coord

        return (
            int(x),
            int(y),
            int(w),
            int(h),
        )

    def _build_track_map(
        self,
        frame_matches_l,
    ):
        """
        Tworzy mapowanie:

            bbox -> track_id
        """

        track_map = {}

        for (
            track_id,
            previous_bb,
            current_bb,
        ) in frame_matches_l:

            if current_bb is None:
                continue

            key = self._bbox_key(
                current_bb
            )

            track_map[key] = track_id

        return track_map


    def _get_track_id(
        self,
        desc,
        track_map,
    ):
        """
        Próbuje przypisać descriptor do track_id
        na podstawie bboxa lewej kamery.
        """

        if not hasattr(
            desc,
            "coordL"
        ):
            return None

        if desc.coordL is None:
            return None

        x, y, w, h = desc.coordL

        descriptor_key = (
            int(x),
            int(y),
            int(w),
            int(h),
        )

        if descriptor_key in track_map:

            return track_map[
                descriptor_key
            ]

        center_x = x + w / 2.0
        center_y = y + h / 2.0

        best_id = None
        best_distance = float("inf")

        for (
            track_key,
            track_id,
        ) in track_map.items():

            bx, by, bw, bh = track_key

            bbox_center_x = (
                bx + bw / 2.0
            )

            bbox_center_y = (
                by + bh / 2.0
            )

            distance = (
                (center_x - bbox_center_x) ** 2
                +
                (center_y - bbox_center_y) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_id = track_id

        return best_id

    def _draw_distances(
        self,
        image,
        objects,
    ):

        output = image.copy()

        for obj in objects.values():

            x = obj["bbox"]["x"]
            y = obj["bbox"]["y"]
            w = obj["bbox"]["w"]
            h = obj["bbox"]["h"]

            distance = obj["distance"]

            cv2.rectangle(
                output,
                (
                    x,
                    y,
                ),
                (
                    x + w,
                    y + h,
                ),
                (0, 255, 0),
                2,
            )


            if distance is None:

                text = (
                    f"{obj['class']} "
                    f"N/A"
                )

            else:

                text = (
                    f"{obj['class']} "
                    f"{distance:.2f} m"
                )

            cv2.putText(
                output,
                text,
                (
                    x,
                    max(
                        20,
                        y - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    def _draw_ids(
        self,
        image,
        objects,
    ):

        output = image.copy()

        for obj in objects.values():

            x = obj["bbox"]["x"]
            y = obj["bbox"]["y"]
            w = obj["bbox"]["w"]
            h = obj["bbox"]["h"]

            track_id = obj["id"]

            cv2.rectangle(
                output,
                (
                    x,
                    y,
                ),
                (
                    x + w,
                    y + h,
                ),
                (255, 0, 0),
                2,
            )

            text = (
                f"ID: {track_id}"
            )

            cv2.putText(
                output,
                text,
                (
                    x,
                    max(
                        20,
                        y - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    def pipeline(
        self,
        frameL_path: str,
        frameR_path: str,
        previous_boxes_l=None,
    ):
        """
        Główny pipeline:

            YOLO
              ↓
            Tracking
              ↓
            Stereo BB matching
              ↓
            Point matching
              ↓
            Triangulation
              ↓
            Kalman
              ↓
            Visualization

        Returns
        -------
        distance_image:
            Obraz z BB + klasą + odległością.

        id_image:
            Obraz z BB + track ID.

        objects:
            Dictionary zawierający dane każdego obiektu.
        """


        framel, boxes_l = (
            self.inference.pipeline(
                frameL_path,
                conf=0.2,
                iou=0.4,
            )
        )

        framer, boxes_r = (
            self.inference.pipeline(
                frameR_path,
                conf=0.2,
                iou=0.4,
            )
        )

        if previous_boxes_l is not None:

            frame_matches_l = (
                self.frameMatcher.pipeline(
                    previous_boxes_l,
                    boxes_l,
                )
            )

        else:

            frame_matches_l = []

            for bb in boxes_l:

                track_id = (
                    self.frameMatcher._create_track(
                        bb
                    )
                )

                frame_matches_l.append(
                    (
                        track_id,
                        None,
                        bb,
                    )
                )

        track_map = (
            self._build_track_map(
                frame_matches_l
            )
        )

        pairs = self.bbSearcher.pipeline(
            boxes_l,
            boxes_r,
            framel,
            framer,
        )

        vis_frame, descriptors = (
            self.ptrMatcher.findPoints(
                pairs,
                framel,
                framer,
            )
        )


        for desc in descriptors:

            self.tranPkt.process_descriptor(
                desc
            )


        objects = {}

        for desc in descriptors:

            track_id = (
                self._get_track_id(
                    desc,
                    track_map,
                )
            )

            if track_id is None:
                continue

            measured_z = (
                desc.triangulation_value
            )

            filtered_z = (
                self._update_kalman(
                    track_id,
                    measured_z,
                )
            )

            x, y, w, h = (
                map(
                    int,
                    desc.coordL,
                )
            )


            cls = getattr(
                desc,
                "cls",
                "unknown",
            )

            objects[track_id] = {

                "id": track_id,

                "class": cls,

                "bbox": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                },

                "distance": filtered_z,
            }

        for (
            track_id,
            previous_bb,
            current_bb,
        ) in frame_matches_l:

            if current_bb is None:
                continue

            if track_id in objects:
                continue

            x, y, w, h = (
                map(
                    int,
                    current_bb.coord,
                )
            )

            cls = getattr(
                current_bb,
                "cls",
                "unknown",
            )

            predicted_z = (
                self._update_kalman(
                    track_id,
                    None,
                )
            )

            objects[track_id] = {

                "id": track_id,

                "class": cls,

                "bbox": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                },

                "distance": predicted_z,
            }

        distance_image = (
            self._draw_distances(
                framel,
                objects,
            )
        )

        id_image = (
            self._draw_ids(
                framel,
                objects,
            )
        )

        return (
            distance_image,
            id_image,
            objects,
        )
