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

# Inicjalizacja komponentów
inference = Inference("models_weights/yolo11n.pt")
bbSearcher = BBSearcher(threshold=1000)
ptrMacher = PointMatcher()
tranPkt = TriangulationPkt()

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
    framel, boxes_and_classes_l = inference.pipeline(i_l, conf=0.2)
    framer, boxes_and_classes_r = inference.pipeline(i_r, conf=0.2)

    # 1. Wyszukanie par bounding boxów
    pairs = bbSearcher.pipeline(
        boxes_and_classes_l, boxes_and_classes_r, framel, framer
    )

    # 2. Wyszukanie punktów charakterystycznych wewnątrz bounding boxów
    ptframe, pts = ptrMacher.findPoints(pairs, framel, framer)

    # ==========================================
    # 3. TRIANGULACJA (OBLICZANIE GŁĘBOKOŚCI Z)
    # ==========================================
    all_pts_l, all_pts_r = pts  # Rozpakowanie punktów z krotki
    
    for pts_l, pts_r in zip(all_pts_l, all_pts_r):
        if len(pts_l) > 0:
            # Uruchomienie metody pipeline z TriangulationPkt dla zestawu punktów 
            z_distance = tranPkt.pipeline(pts_l, pts_r)
            
            print(f"Znaleziono obiekt -> szacowana głębokość (Z): {z_distance:.2f} m")
            
            # Wypisanie wyniku (w metrach) na lewej części wizualizacji (ptframe)
            if z_distance > 0:
                # Bierzemy współrzędne pierwszego punktu z zestawu, aby mieć punkt zaczepienia do rysowania
                px, py = int(pts_l[0][0]), int(pts_l[0][1])
                cv2.putText(
                    ptframe,
                    f"Z: {z_distance:.2f}m",
                    (px, max(25, py - 15)), # Unikamy rysowania poza górną krawędzią obrazu
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
    # ==========================================

    img_pairs = plotBBPairs(framel, framer, pairs)

    if len(img_pairs.shape) == 3 and img_pairs.shape[2] == 3:
        pass

    box_frame_l = inference.plot_bounding_box(i_l)
    box_frame_r = inference.plot_bounding_box(i_r)
    img_boxes = cv2.hconcat([box_frame_l, box_frame_r])

    target_width = ptframe.shape[1]

    def resize_to_width(img, width):
        if img.shape[1] != width:
            height = int(img.shape[0] * (width / img.shape[1]))
            return cv2.resize(img, (width, height))
        return img

    img_pairs = resize_to_width(img_pairs, target_width)
    img_boxes = resize_to_width(img_boxes, target_width)

    final_debug_view = cv2.vconcat([img_pairs, img_boxes, ptframe])

    display_height = 900
    if final_debug_view.shape[0] > display_height:
        scale = display_height / final_debug_view.shape[0]
        final_debug_view = cv2.resize(final_debug_view, (0, 0), fx=scale, fy=scale)

    cv2.imshow("Stereo Bounding Box & Point Matching", final_debug_view)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()