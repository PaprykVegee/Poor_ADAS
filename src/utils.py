import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.yoloEval import *


def plotBBPairs(
    imgl: np.ndarray, 
    imgr: np.ndarray, 
    bbs: list[tuple[BoxDesc, BoxDesc]]
) -> np.ndarray:
    imgl_draw = imgl.copy()
    imgr_draw = imgr.copy()

    for left_bb, right_bb in bbs:  
        lx, ly, lw, lh = left_bb.coord
        rx, ry, rw, rh = right_bb.coord

        color = tuple(int(c) for c in np.random.randint(0, 256, size=3))

        cv2.rectangle(
            imgl_draw, (lx, ly), (lx + lw, ly + lh), color=color, thickness=2
        )
        cv2.rectangle(
            imgr_draw, (rx, ry), (rx + rw, ry + rh), color=color, thickness=2
        )

    return cv2.hconcat([imgl_draw, imgr_draw])