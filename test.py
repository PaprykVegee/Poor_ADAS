import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.characteristicPointMatcher import *
from src.localBoundingBoxSearch import *
from src.utils import *
from src.yoloEval import *
from src.triangulationCharacteristicsPoint import *
from src.objectCentricStereo import *

# # Inicjalizacja komponentów
# inference = Inference("models_weights/yolo11n.pt")
# bbSearcher = BBSearcher(threshold=1000)
# ptrMacher = PointMatcher()
# tranPkt = TriangulationPkt()

# valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# left_images = sorted([
#     os.path.join(root, file)
#     for root, _, files in os.walk("left_img_folder")
#     for file in files
#     if file.lower().endswith(valid_extensions)
# ])[:100]

# right_images = sorted([
#     os.path.join(root, file)
#     for root, _, files in os.walk("right_img_folder")
#     for file in files
#     if file.lower().endswith(valid_extensions)
# ])[:100]

# for i_l, i_r in zip(left_images, right_images):
#     framel, boxes_and_classes_l = inference.pipeline(i_l, conf=0.2, iou=0.4)
#     framer, boxes_and_classes_r = inference.pipeline(i_r, conf=0.2, iou=0.4)

#     pairs = bbSearcher.pipeline(
#         boxes_and_classes_l, boxes_and_classes_r, framel, framer
#     )

#     ptframe, pts = ptrMacher.findPoints(pairs, framel, framer)

#     img_pairs = plotBBPairs(framel, framer, pairs)

#     if len(img_pairs.shape) == 3 and img_pairs.shape[2] == 3:
#         pass

#     box_frame_l = inference.plot_bounding_box(i_l)
#     box_frame_r = inference.plot_bounding_box(i_r)
#     img_boxes = cv2.hconcat([box_frame_l, box_frame_r])

#     target_width = ptframe.shape[1]

#     def resize_to_width(img, width):
#         if img.shape[1] != width:
#             height = int(img.shape[0] * (width / img.shape[1]))
#             return cv2.resize(img, (width, height))
#         return img

#     img_pairs = resize_to_width(img_pairs, target_width)
#     img_boxes = resize_to_width(img_boxes, target_width)

#     final_debug_view = cv2.vconcat([img_pairs, img_boxes, ptframe])

#     display_height = 900
#     if final_debug_view.shape[0] > display_height:
#         scale = display_height / final_debug_view.shape[0]
#         final_debug_view = cv2.resize(final_debug_view, (0, 0), fx=scale, fy=scale)

#     cv2.imshow("Stereo Bounding Box & Point Matching", final_debug_view)

#     key = cv2.waitKey(0) & 0xFF
#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()


import os
import cv2
import numpy as np


import os
import cv2


def main():
    ocs = ObjectCentricStereo()

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    left_images = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk("left_img_folder")
        for file in files
        if file.lower().endswith(valid_extensions)
    ])[100:400]

    right_images = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk("right_img_folder")
        for file in files
        if file.lower().endswith(valid_extensions)
    ])[100:400]

    for i_l, i_r in zip(left_images, right_images):
        # 1. Obliczamy wyniki z stereo pipeline
        vis_frame, descriptors = ocs.pipeline(i_l, i_r)

        # 2. Wczytujemy czyste, oryginalne zdjęcie z lewej kamery
        left_img = Inference.imread_rgb(i_l)
        # Konwersja na BGR dla OpenCV (do wyświetlenia/zapisu)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)

        # 3. Rysujemy ramki i odległości WEWNĄTRZ ramki na zdjęciu
        for desc in descriptors:
            if desc.coordL is None:
                continue

            lx, ly, lw, lh = map(int, desc.coordL)
            z_val = desc.triangulation_value

            label = f"{desc.cls}: {z_val:.2f}m" if z_val is not None else f"{desc.cls}: N/A"

            # Rysowanie zielonej ramki wokół obiektu
            cv2.rectangle(left_img, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)

            # Obliczanie pozycji tekstu, aby znalazł się WEWNĄTRZ ramki
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Próba umieszczenia w górnej części wewnątrz ramki
            text_x = lx + 5
            text_y = ly + text_h + 8

            # Zabezpieczenie przed wyjściem tekstu na dole, jeśli BB jest bardzo mały
            if text_y > ly + lh:
                text_y = ly + lh - 5

            # Czarny kwadrat pod tekst (wewnątrz BB) dla czytelności
            cv2.rectangle(
                left_img,
                (text_x - 2, text_y - text_h - 4),
                (text_x + text_w + 2, text_y + 4),
                (0, 0, 0),
                -1
            )

            # Nanoszenie tekstu wewnątrz ramki
            cv2.putText(
                left_img,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        # 4. Wyświetlanie wyników w dwóch osobnych oknach
        cv2.imshow("1. Visualisation with Matching Lines", vis_frame)
        cv2.imshow("2. Left Frame - Bounding Boxes & Distance Inside", left_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
