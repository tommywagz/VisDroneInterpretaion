import torch
import yaml
from pathlib import Path

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import check_img_size, increment_path
from utils.torch_utils import select_device
from utils.loss import ComputeLoss
from utils.plots import plot_images

# Setup
device = select_device('0')  # Use RTX 2060
batch_size = 16  # Adjust based on memory
imgsz = 640  # YOLO input size
epochs = 100

# 1. Initialize model
model = Model(cfg='models/yolov5s.yaml', ch=3, nc=10).to(device)  # Create YOLOv5-small with 10 classes

# 2. Create VisDrone dataset YAML configuration
visdrone_yaml = {
    # 'path': './VisDrone2019',  # Dataset root directory
    'train': 'VisDrone2019-DET-train/images',
    'val': 'images',

    'nc': 10,  # Number of classes
    'names': [
        'pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
}

# Save YAML config
with open('data/visdrone.yaml', 'w') as f:
    yaml.dump(visdrone_yaml, f)

# 3. Modify anchors for small objects
anchors = [[10,13, 16,30, 33,23],  # P3/8
           [30,61, 62,45, 59,119],  # P4/16  
           [116,90, 156,198, 373,326]]  # P5/32
model.yaml['anchors'] = anchors

# 4. Custom dataloader to handle VisDrone annotations
def visdrone_to_yolo(visdrone_anno_path, output_label_path):
    """Convert VisDrone annotation format to YOLO format"""
    with open(visdrone_anno_path, 'r') as f:
        lines = f.readlines()
    
    yolo_annotations = []
    for line in lines:
        bbox = line.strip().split(',')
        x, y, w, h, score, class_id, _, _ = map(int, bbox[:8])
        if class_id == 0 or class_id > 10:  # Skip 'ignored regions' (class 0)
            continue
            
        # Convert to YOLO format: class_id x_center y_center width height
        img_w, img_h = 1920, 1080  # Typical VisDrone image size, adjust if needed
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h
        
        # Convert class_id (1-10 in VisDrone) to 0-9 for YOLO
        yolo_annotations.append(f"{class_id-1} {x_center} {y_center} {width} {height}")
    
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

# 5. Data augmentation pipeline focused on small objects
augment = True
hyp = {  # Hyperparameters
    'lr0': 0.01,
    'lrf': 0.1,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05,
    'cls': 0.5,
    'cls_pw': 1.0,
    'obj': 1.0,
    'obj_pw': 1.0,
    'iou_t': 0.20,
    'anchor_t': 4.0,
    'fl_gamma': 0.0,
    'hsv_h': 0.015,  # HSV augmentation
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,  # Rotation 
    'translate': 0.1,  # Translation
    'scale': 0.5,  # Scale up to focus on small objects
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,  # Mosaic augmentation
    'mixup': 0.0,  # Mixup augmentation
}

# 6. Train model
import train
results = train.train(
    hyp=hyp,
    data='data/visdrone.yaml',
    imgsz=640,
    epochs=epochs,
    batch_size=batch_size,
    weights='yolov5s.pt',  # Transfer learning from COCO
    workers=4,
    device=device,
    mixed_precision=True,
    accumulate=4
)

model.eval()
inference_size = 512  # Lower for faster inference

import os
import glob
from pathlib import Path
import shutil
import concurrent.futures

def process_dataset(src_img_dir, src_anno_dir, dst_dir, split):
    """Process and convert VisDrone dataset to YOLO format"""
    # Create directories
    images_dir = os.path.join(dst_dir, split, 'images')
    labels_dir = os.path.join(dst_dir, split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get image files
    img_files = glob.glob(os.path.join(src_img_dir, '*.jpg'))
    
    def process_file(img_path):
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        # Copy image
        dst_img_path = os.path.join(images_dir, img_name)
        shutil.copy(img_path, dst_img_path)
        
        # Convert annotation
        anno_path = os.path.join(src_anno_dir, f"{img_stem}.txt")
        if os.path.exists(anno_path):
            dst_label_path = os.path.join(labels_dir, f"{img_stem}.txt")
            visdrone_to_yolo(anno_path, dst_label_path)
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_file, img_files))

# Usage
process_dataset(
    'VisDrone2019-DET-train/images',
    'VisDrone2019-DET-train/annotations',
    'datasets/visdrone-yolo',
    'train'
)

from utils.autoanchor import kmean_anchors
kmean_anchors(dataset='data/visdrone.yaml', n=9, img_size=640)