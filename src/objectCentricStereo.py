import cv2
import numpy as np
from src.characteristicPointMatcher import PointMatcher, ChPtrDesc
from src.localBoundingBoxSearch import BBSearcher
from src.triangulationCharacteristicsPoint import TriangulationPkt
from src.yoloEval import Inference
from src.BBMatcher import BBLeftRightMatcher


class ObjectCentricStereo:
    def __init__(
        self,
        model_path: str = "models_weights/yolo11n.pt",
        threshold: float = 500.0,
        max_y_diff: float = 50.0,
        corr_weight: float = 300.0,
        lowa_ratio: float = 0.9,
        baseline: float = 0.5,
    ) -> None:
        self.inference = Inference(model_path)
        self.bbSearcher = BBLeftRightMatcher(threshold=threshold, corr_weight=corr_weight)
        self.ptrMatcher = PointMatcher(ratio_thresh=lowa_ratio)
        self.tranPkt = TriangulationPkt(baseline=baseline)

    def pipeline(self, frameL_path: str, frameR_path: str) -> tuple[np.ndarray, list[ChPtrDesc]]:
        framel, boxes_and_classes_l = self.inference.pipeline(frameL_path, conf=0.2, iou=0.4)
        framer, boxes_and_classes_r = self.inference.pipeline(frameR_path, conf=0.2, iou=0.4)

        pairs = self.bbSearcher.pipeline(boxes_and_classes_l, boxes_and_classes_r, framel, framer)

        vis_frame, descriptors = self.ptrMatcher.findPoints(pairs, framel, framer)

        for desc in descriptors:
            self.tranPkt.process_descriptor(desc)

        return vis_frame, descriptors