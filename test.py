import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.yoloEval import *
from src.localBoundingBoxSearch import *

import os
from pathlib import Path

inference = Inference("models_weights/yolo11n.pt")
bbSearcher = BBSearcher(threshold=50)

valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

left_images = [
    os.path.join(root, file)
    for root, _, files in os.walk("left_img_folder")
    for file in files
    if file.lower().endswith(valid_extensions)
][:100]

right_images = [
    os.path.join(root, file)
    for root, _, files in os.walk("right_img_folder")
    for file in files
    if file.lower().endswith(valid_extensions)
][:100]

for i_l, i_r in zip(left_images, right_images):
    framel, boxes_and_classes_l = inference.pipeline(i_l)
    framer, boxes_and_classes_r = inference.pipeline(i_r)

    pairs = bbSearcher.pipeline(boxes_and_classes_l, boxes_and_classes_r)

    img = plotBBPairs(framel, framer, pairs)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Stereo Bounding Box Matching", img_bgr)

    # POPRAWKA 3: Zmień na 0, jeśli chcesz zatrzymywać każdą klatkę do wciśnięcia klawisza
    if cv2.waitKey(0) & 0xFF == ord("q"):  # Naciśnij 'q' aby przerwać
        break

cv2.destroyAllWindows()