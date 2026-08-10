import os
import cv2
import numpy as np

import time

from src.yoloEval import Inference
from src.BBMatcher import (
    BBLeftRightMatcher,
    FrameMatcher,
)
from src.characteristicPointMatcher import PointMatcher
from src.triangulationCharacteristicsPoint import TriangulationPkt


def draw_tracking_info(
    image: np.ndarray,
    tracks: list,
) -> np.ndarray:

    output = image.copy()

    for track_id, left_bb, right_bb in tracks:

        x, y, w, h = map(
            int,
            left_bb.coord,
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        label = f"ID: {track_id}"

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )

        text_x = x + 5
        text_y = y + text_h + 5

        cv2.rectangle(
            output,
            (
                text_x - 2,
                text_y - text_h - 4,
            ),
            (
                text_x + text_w + 2,
                text_y + 4,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_distance(
    image: np.ndarray,
    descriptors,
) -> np.ndarray:

    output = image.copy()

    for desc in descriptors:

        if desc.coordL is None:
            continue

        lx, ly, lw, lh = map(
            int,
            desc.coordL,
        )

        z_val = desc.triangulation_value

        if z_val is not None:
            label = (
                f"{desc.cls}: "
                f"{z_val:.2f} m"
            )
        else:
            label = (
                f"{desc.cls}: N/A"
            )

        cv2.rectangle(
            output,
            (
                lx,
                ly,
            ),
            (
                lx + lw,
                ly + lh,
            ),
            (0, 255, 0),
            2,
        )

        (
            text_w,
            text_h,
        ), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )

        text_x = lx + 5
        text_y = ly + text_h + 8

        if text_y > ly + lh:
            text_y = ly + lh - 5

        cv2.rectangle(
            output,
            (
                text_x - 2,
                text_y - text_h - 4,
            ),
            (
                text_x + text_w + 2,
                text_y + 4,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                text_x,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def main():

    inference = Inference(
        "models_weights/yolo11n.pt"
    )


    stereo_matcher = BBLeftRightMatcher(
        threshold=1000.0,
        local_search_x=1.0,
        local_search_y=0.5,
        corr_weight=300.0,
    )

    frame_matcher = FrameMatcher(
        threshold=300.0,
        local_search_x=3.0,
        local_search_y=2.0,
        position_weight=1.0,
        size_weight=100.0,
    )

    point_matcher = PointMatcher()

    triangulation = TriangulationPkt()


    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    )

    left_images = sorted(
        [
            os.path.join(
                root,
                file,
            )
            for root, _, files in os.walk(
                "left_img_folder"
            )
            for file in files
            if file.lower().endswith(
                valid_extensions
            )
        ]
    )[900:1100]

    right_images = sorted(
        [
            os.path.join(
                root,
                file,
            )
            for root, _, files in os.walk(
                "right_img_folder"
            )
            for file in files
            if file.lower().endswith(
                valid_extensions
            )
        ]
    )[900:1100]

    previous_left_bbs = []
    previous_tracks = []


    for frame_idx, (i_l, i_r) in enumerate(
        zip(
            left_images,
            right_images,
        ),
        start=900,
    ):


        frame_left, left_bbs = inference.pipeline(
            i_l,
            conf=0.2,
            iou=0.4,
        )

        frame_right, right_bbs = inference.pipeline(
            i_r,
            conf=0.2,
            iou=0.4,
        )

        stereo_pairs = stereo_matcher.pipeline(
            left_bbs,
            right_bbs,
            frame_left,
            frame_right,
        )

        current_left_bbs = [
            left_bb
            for left_bb, right_bb
            in stereo_pairs
        ]

        if len(previous_left_bbs) == 0:

            temporal_tracks = []

            for left_bb, right_bb in stereo_pairs:

                track_id = (
                    frame_matcher._create_track(
                        left_bb
                    )
                )

                temporal_tracks.append(
                    (
                        track_id,
                        left_bb,
                        right_bb,
                    )
                )

        else:

            temporal_matches = (
                frame_matcher.pipeline(
                    previous_left_bbs,
                    current_left_bbs,
                )
            )

            temporal_tracks = []

            for (
                track_id,
                previous_bb,
                current_bb,
            ) in temporal_matches:

                for (
                    left_bb,
                    right_bb,
                ) in stereo_pairs:

                    if left_bb is current_bb:

                        temporal_tracks.append(
                            (
                                track_id,
                                left_bb,
                                right_bb,
                            )
                        )

                        break


        descriptors = []

        for (
            track_id,
            left_bb,
            right_bb,
        ) in temporal_tracks:

            pairs = [
                (
                    left_bb,
                    right_bb,
                )
            ]

            try:

                _, descs = (
                    point_matcher.findPoints(
                        pairs,
                        frame_left,
                        frame_right,
                    )
                )

            except Exception as e:

                print(
                    f"[Frame {frame_idx}] "
                    f"Point matching error: {e}"
                )

                continue

            if descs is None:
                continue

            for desc in descs:

                desc = (
                    triangulation.process_descriptor(
                        desc
                    )
                )

                # Zachowujemy ID obiektu.
                desc.track_id = track_id

                descriptors.append(
                    desc
                )

        left_visualization = (
            draw_tracking_info(
                frame_left,
                temporal_tracks,
            )
        )

        distance_visualization = (
            draw_distance(
                left_visualization,
                descriptors,
            )
        )


        stereo_visualization = (
            frame_left.copy()
        )

        for (
            track_id,
            left_bb,
            right_bb,
        ) in temporal_tracks:

            x, y, w, h = map(
                int,
                left_bb.coord,
            )

            cv2.rectangle(
                stereo_visualization,
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

            cv2.putText(
                stereo_visualization,
                f"ID {track_id}",
                (
                    x,
                    max(20, y - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )


        cv2.imshow(
            "Tracking + Distance",
            distance_visualization[:, :, ::-1]
        )

        cv2.imshow(
            "Stereo / Track IDs",
            stereo_visualization[:, :, ::-1]
        )

        key = cv2.waitKey(10) & 0xFF
        time.sleep(0.4)

        if key == ord("q"):
            break


        previous_left_bbs = (
            current_left_bbs.copy()
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()