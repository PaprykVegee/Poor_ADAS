import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


class BoxDesc:

    def __init__(self, coord: list, cls: str, label: int, conf: float):
        self.coord = coord
        self.cls = cls
        self.label = label
        self.conf = conf


class Inference:

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def pipeline(self, image_path: str, conf: float = 0.35, iou: float = 0.35):
        frame = self.imread_rgb(image_path)

        results = self.model(frame, conf=conf, iou=iou, verbose=False)[0]

        boxes_and_classes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            left = int(x1)
            top = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            confidence = float(box.conf[0].item())

            boxes_and_classes.append(
                BoxDesc(
                    [left, top, width, height], cls_name, cls_id, confidence
                )
            )

        return frame, boxes_and_classes

    # POPRAWKA 1: Usunięto 'self'
    @staticmethod
    def plot(img: np.ndarray) -> None:
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    # POPRAWKA 2: Usunięto 'self'
    @staticmethod
    def imread_rgb(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Nie udało się wczytać obrazu: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def plot_bounding_box(self, image_path: str):
        frame, boxes_and_classes = self.pipeline(image_path)

        for bac in boxes_and_classes:
            x, y, w, h = bac.coord
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label_text = f"{bac.cls} {bac.conf:.2f}"
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        self.plot(frame)