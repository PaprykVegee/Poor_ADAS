import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
import random
import shutil
import yaml
import numpy as np
from pathlib import Path
import cv2

class json2yaml:
    def __init__(self, img_path, label_path):
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        self.df = pd.read_json(label_path)
        
        self.img_dict = {}
        for root, dirs, files in os.walk(img_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.img_dict[file] = os.path.join(root, file)

        self.img_name = list(self.img_dict.keys())
        
        all_categories = set()
        for item in self.df['labels']:
            for label in item:
                if 'box2d' in label:
                    all_categories.add(label['category'])
        print(all_categories)
                
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(all_categories))
        
    def create_txt_cord(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        for name in tqdm(self.img_name):
            sample = self.df[self.df['name'] == name]
            if sample.empty:
                continue
            
            folder = 'train' if random.random() < 0.8 else 'val'
            img_dir = os.path.join(output_dir, folder, 'images')
            os.makedirs(img_dir, exist_ok=True)
            
            txt_file_path = os.path.join(
                output_dir,
                folder,
                'labels',
                name.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
            
            img_full_path = self.img_dict.get(name)
            if img_full_path:
                shutil.copy(img_full_path, img_dir)
            
            with open(txt_file_path, 'w') as f:
                for row in sample['labels'].iloc[0]:
                    if 'box2d' in row:
                        box = row['box2d']
                        x1, y1 = int(box['x1']), int(box['y1'])
                        x2, y2 = int(box['x2']), int(box['y2'])
                        
                        try:
                            image = Image.open(img_full_path)
                            width, height = image.size
                        except Exception as e:
                            print(f"Cannot open image {name}: {e}")
                            continue
                        
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        category = row['category']
                        category_id = self.label_encoder.transform([category])[0]
                        
                        f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
    
    def create_yaml(self, output_yaml_path):
        yaml_data = {
            'train': os.path.join(self.output_dir, "train"),
            'val': os.path.join(self.output_dir, "val"),
            'nc': len(self.label_encoder.classes_),
            'names': self.label_encoder.classes_.tolist()
        }
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)

    def decode_label(self, label_id: np.array):
        return self.label_encoder.inverse_transform(label_id)
    

from sklearn.model_selection import train_test_split


class txt2yaml:
    def __init__(self, folder_images: Path, folder_labels: Path) -> None:
        self.dic = {}
        for label_file in folder_labels.iterdir():
            if label_file.suffix == '.txt':
                self.dic[label_file.stem] = label_file

        new_dic = {}
        for image_file in folder_images.iterdir():
            if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                id = image_file.stem
                if id in self.dic:
                    new_dic[image_file] = self.dic[id]
        self.dic = new_dic

    def create_txt(self, path: str, name: str = 'kitty', test_size: float = 0.2, random_state: int = 42):
        folder_name = os.path.join(path, name)
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(os.path.join(folder_name, 'train', 'images'))
        os.makedirs(os.path.join(folder_name, 'train', 'labels'))
        os.makedirs(os.path.join(folder_name, 'test', 'images'))
        os.makedirs(os.path.join(folder_name, 'test', 'labels'))

        train_path_images = os.path.join(folder_name, 'train', 'images')
        train_path_labels = os.path.join(folder_name, 'train', 'labels')
        test_path_images = os.path.join(folder_name, 'test', 'images')
        test_path_labels = os.path.join(folder_name, 'test', 'labels')

        keys = list(self.dic.keys())
        train_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=random_state)

        train_dic = {k: self.dic[k] for k in train_keys}
        test_dic = {k: self.dic[k] for k in test_keys}

        # train
        for img, label in tqdm(train_dic.items()):
            shutil.copy(img, os.path.join(train_path_images, img.name))
            self._convert_label(img, label, os.path.join(train_path_labels, label.name))

        # test
        for img, label in tqdm(test_dic.items()):
            shutil.copy(img, os.path.join(test_path_images, img.name))
            self._convert_label(img, label, os.path.join(test_path_labels, label.name))

    def _convert_label(self, img_path: Path, label_path: Path, out_path: str):
        # odczyt rozmiaru obrazu
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape

        converted = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                class_name = parts[0]
                if class_name.lower() == "dontcare":
                    continue

                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])

                cx = ((xmin + xmax) / 2) / w
                cy = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                converted.append(f"{class_name} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(converted) + "\n")
