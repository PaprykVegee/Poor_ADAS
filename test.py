
import os
import cv2
import time

from src.objectCentricStereo import ObjectCentricStereo


def main():

    # ==================================================
    # OBJECT CENTRIC STEREO
    # ==================================================

    stereo = ObjectCentricStereo(
        model_path="models_weights/yolo11n.pt",

        # Stereo
        threshold=500.0,
        corr_weight=300.0,
        lowa_ratio=0.9,
        baseline=0.5,

        # Tracking
        max_age=5,

        # Kalman
        kalman_dt=1.0,
        kalman_process_noise=0.01,
        kalman_measurement_noise=1.0,
    )

    # ==================================================
    # IMAGE EXTENSIONS
    # ==================================================

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    )

    # ==================================================
    # LEFT IMAGES
    # ==================================================

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
    )

    # ==================================================
    # RIGHT IMAGES
    # ==================================================

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
    )

    # ==================================================
    # CHECK
    # ==================================================

    if len(left_images) == 0:

        print(
            "ERROR: No left images found."
        )

        return

    if len(right_images) == 0:

        print(
            "ERROR: No right images found."
        )

        return

    if len(left_images) != len(right_images):

        print(
            "WARNING:"
        )

        print(
            f"Left images : {len(left_images)}"
        )

        print(
            f"Right images: {len(right_images)}"
        )

        print(
            "Using only the common number of frames."
        )

    number_of_frames = min(
        len(left_images),
        len(right_images),
    )

    print(
        f"Frames: {number_of_frames}"
    )

    # ==================================================
    # TRACKING STATE
    # ==================================================

    previous_boxes_l = None

    # ==================================================
    # MAIN LOOP
    # ==================================================

    for frame_idx in range(
        number_of_frames
    ):

        left_path = left_images[
            frame_idx
        ]

        right_path = right_images[
            frame_idx
        ]

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"FRAME {frame_idx}"
        )

        print(
            f"LEFT : {left_path}"
        )

        print(
            f"RIGHT: {right_path}"
        )

        print(
            "=" * 60
        )

        # ==================================================
        # PIPELINE
        # ==================================================

        try:

            (
                distance_image,
                id_image,
                objects,
            ) = stereo.pipeline(
                left_path,
                right_path,
                previous_boxes_l,
            )

        except Exception as e:

            print(
                f"Pipeline error: {e}"
            )

            import traceback

            traceback.print_exc()

            continue

        # ==================================================
        # PRINT OBJECT DATA
        # ==================================================

        print(
            f"\nDetected objects: "
            f"{len(objects)}"
        )

        for (
            track_id,
            obj,
        ) in objects.items():

            bbox = obj["bbox"]

            distance = obj["distance"]

            print(
                f"\nID: {track_id}"
            )

            print(
                f"Class: {obj['class']}"
            )

            print(
                "BBox:"
                f" x={bbox['x']}"
                f" y={bbox['y']}"
                f" w={bbox['w']}"
                f" h={bbox['h']}"
            )

            if distance is None:

                print(
                    "Distance: N/A"
                )

            else:

                print(
                    f"Distance: "
                    f"{distance:.3f} m"
                )

        # ==================================================
        # DISPLAY
        # ==================================================

        cv2.imshow(
            "Object Centric Stereo - Distance",
            distance_image[:, :, ::-1],
        )

        cv2.imshow(
            "Object Centric Stereo - IDs",
            id_image[:, :, ::-1],
        )

        # ==================================================
        # NEXT FRAME
        #
        # YOLO boxes from the current frame
        # are needed by FrameMatcher.
        #
        # We can obtain them directly from
        # the objects returned by pipeline.
        # ==================================================

        previous_boxes_l = []

        for obj in objects.values():

            bbox = obj["bbox"]

            previous_boxes_l.append(
                {
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "w": bbox["w"],
                    "h": bbox["h"],
                }
            )

        # ==================================================
        # WAIT
        # ==================================================

        cv2.waitKey(0)
        #time.sleep(0.2)

        # if key == ord("q"):

        #     print(
        #         "Exiting..."
        #     )

        #     break

        # Opcjonalnie:
        # time.sleep(0.4)

    # ==================================================
    # CLEANUP
    # ==================================================

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
