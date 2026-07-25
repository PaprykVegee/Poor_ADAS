import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.characteristicPointMatcher import *
from src.localBoundingBoxSearch import *
from src.utils import *
from src.yoloEval import *

# Inicjalizacja komponentów
inference = Inference("models_weights/yolo11n.pt")
bbSearcher = BBSearcher(threshold=1000)
ptrMacher = PointMatcher()

valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

left_images = sorted([
    os.path.join(root, file)
    for root, _, files in os.walk("left_img_folder")
    for file in files
    if file.lower().endswith(valid_extensions)
])[:100]

right_images = sorted([
    os.path.join(root, file)
    for root, _, files in os.walk("right_img_folder")
    for file in files
    if file.lower().endswith(valid_extensions)
])[:100]

for i_l, i_r in zip(left_images, right_images):
    # 1. Detekcja YOLO dla obu kamer
    framel, boxes_and_classes_l = inference.pipeline(i_l, conf=0.2)
    framer, boxes_and_classes_r = inference.pipeline(i_r, conf=0.2)

    # 2. Szukanie pasujących ramek BB
    pairs = bbSearcher.pipeline(
        boxes_and_classes_l, boxes_and_classes_r, framel, framer
    )

    # 3. Wyznaczanie punktów charakterystycznych wewnątrz ramek
    # ROZPAKOWANIE KROTKI: findPoints zwraca (vis_frame, (all_ptsL, all_ptsR))
    ptframe, pts = ptrMacher.findPoints(pairs, framel, framer)

    # 4. Rysowanie dopasowań ramek (wizualizacja z utils)
    img_pairs = plotBBPairs(framel, framer, pairs)

    # Upewnij się czy img_pairs wymaga konwersji RGB -> BGR
    if len(img_pairs.shape) == 3 and img_pairs.shape[2] == 3:
        # Jeśli plotBBPairs używa Matplotlib (RGB), odkomentuj poniższą linię:
        # img_pairs = cv2.cvtColor(img_pairs, cv2.COLOR_RGB2BGR)
        pass

    # 5. Przygotowanie ramek z detekcjami YOLO
    box_frame_l = inference.plot_bounding_box(i_l)
    box_frame_r = inference.plot_bounding_box(i_r)
    img_boxes = cv2.hconcat([box_frame_l, box_frame_r])

    # 6. Zabezpieczenie szerokości przed vconcat (dopasowanie do wymiaru ptframe)
    target_width = ptframe.shape[1]

    def resize_to_width(img, width):
        if img.shape[1] != width:
            height = int(img.shape[0] * (width / img.shape[1]))
            return cv2.resize(img, (width, height))
        return img

    img_pairs = resize_to_width(img_pairs, target_width)
    img_boxes = resize_to_width(img_boxes, target_width)

    # 7. Połączenie w pionie: (Góra: Rysunek par BB | Środek: Detekcje YOLO | Dół: Punkty ORB)
    final_debug_view = cv2.vconcat([img_pairs, img_boxes, ptframe])

    # Opcjonalne skalowanie całego widoku, jeśli obraz nie mieści się na ekranie
    display_height = 900
    if final_debug_view.shape[0] > display_height:
        scale = display_height / final_debug_view.shape[0]
        final_debug_view = cv2.resize(final_debug_view, (0, 0), fx=scale, fy=scale)

    cv2.imshow("Stereo Bounding Box & Point Matching", final_debug_view)

    # Sterowanie: 'q' przerywa, dowolny inny klawisz przechodzi do kolejnej klatki
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()