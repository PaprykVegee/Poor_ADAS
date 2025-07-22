import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

def imread_rgb(path: str) -> np.array:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img

def plot(img: np.array) -> None:
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def plot_rectangle(img, corrd, cls):
    img = cv2.rectangle(img, (corrd[0], corrd[1]), (corrd[0] + corrd[2], corrd[1] + corrd[3]), (0, 255, 255), 2)
    img = cv2.putText(img, cls, (corrd[0], corrd[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img

def plot_prdictions(model, img):
    results = model(img)[0]
    boxes = results.boxes.xyxy  
    confidences = results.boxes.conf 
    classes = results.boxes.cls 

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())  
        class_id = int(classes[i].item()) 
        confidence = confidences[i].item()  
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  

        label = f"Class {[class_id]} ({confidence:.2f})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img


class box_desc:
    def __init__(self, coord: np.array, cls: str, label: int):
        self.coord = coord
        self.cls = cls
        self.label = label

class Inference:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.classes = {
            0: 'bike', 1: 'bus', 2: 'car', 3: 'motor', 4: 'person',
            5: 'rider', 6: 'traffic light', 7: 'traffic sign', 8: 'train', 9: 'truck'
        }

    def __detector(self, image_data):
        return self.session.run(["output0"], {"images": image_data})

    def __postprocessor(self, results, frame_shape, confidence=0.35, iou=0.35):
        img_height, img_width = frame_shape[:2]
        outputs = np.transpose(np.squeeze(results[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= confidence:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou)

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                cls = class_ids[idx]
                results.append(box_desc([box[0], box[1], box[2], box[3]], self.classes[cls], cls))

        return results
    
    def preprocessor(self, frame):
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        image_data = np.array(resized).astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)
        return image_data
    
    def pipeline(self, image_path):
        frame = imread_rgb(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = self.preprocessor(frame)
        results = self.__detector(image_data)
        boxes_and_classes = self.__postprocessor(results, frame.shape)
        return frame, boxes_and_classes
    
    def plot_bounding_box(self, image_path):
        frame, boxes_and_classes = self.pipeline(image_path)

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for bac in boxes_and_classes:
            coord = bac.coord
            cls_name = bac.cls
            cls = bac.cls
            frame = plot_rectangle(frame, coord, cls)

        plot(frame)