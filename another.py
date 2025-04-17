import os, datetime, time, sys
import numpy as np
from PIL import Image
from collections import deque, defaultdict

import torch.nn as nn
import torchvision, torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


######## Process that Data ##########
class VisDroneDataset(Dataset):
    def __init__(self, imagePath, annotPath, transforms=None):
        self.imgDir = imagePath
        self.annotationPath = annotPath
        self.transforms = transforms
        self.imageFiles = sorted(os.listdir(imagePath))
        self.annotations = self.getAnnotations()
        self.classes = ['ignored region', 'pedestrian', 'people', 'car', 'van', 'bus', 'truck', 'tricycle', 'awning-tricycle', 'bicycle', 'motorcycle']
        
    def getAnnotations(self):
        annotations = {}
        for img in self.imageFiles:
            currAnnotationName = img.replace(".jpg", ".txt")
            currAnnotationPath = os.path.join(self.annotationPath, currAnnotationName)
            boxes = []
            labels = []
            
            if os.path.exists(currAnnotationPath):
                with open(currAnnotationPath, 'r') as f:
                    for line in f:
                        try:
                            x, y, w, h, score, category, truncation, occlusion = map(int, line.strip().split(',')[:8])
                            if 1 <= category <= 10 and w > 0 and h > 0:
                                boxes.append([x, y, x + w, y + h])
                                labels.append(category)
            
                        except ValueError as e:
                            print(f"Error parsing line in {currAnnotationPath}: {line.strip()} - {e}")
            
            annotations[img] = {'boxes': boxes, 'labels': labels}
            
        return annotations
                            
                            
    def __len__(self):
        return len(self.imageFiles)
    
    
    def __getitem__(self, idx):
        imgName = self.image_files[idx]
        imgPath = os.path.join(self.root_dir, imgName)
        annotation_data = self.annotations[imgName]

        img = Image.open(imgPath).convert("RGB")
        boxes = []
        labels = []

        for i, box in enumerate(annotation_data['boxes']):
            x_min, y_min, x_max, y_max = box
            if x_min < x_max and y_min < y_max:
                boxes.append(box)
                labels.append(annotation_data['labels'][i])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            # If your transforms modify boxes, ensure they handle potential invalid cases
            transformed = self.transforms(image=np.array(img), bboxes=np.array(boxes), labels=np.array(labels))
            img = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        return img, target
    
    
############ transform the images in the dataset ##############
def get_transform(train):
    transforms = [A.Resize(height=600, width=800)]
    transforms.append(A.ToFloat())
    if train:
        transforms.extend([A.HorizontalFlip(p=0.5), ToTensorV2()])
        # More augmentations can be added here
        # transforms.append(T.RandomPhotometricDistort())
        # transforms.append(T.RandomZoomOut())
        # transforms.append(T.RandomAdjustSharpness(0.3))
    else:
        transforms.append(ToTensorV2())
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# combine the photo and its correlated annotation
def collate_fn(batch):
    return tuple(zip(*batch))
    
    
########### Load the Dataset ############
# paths to datasets
pathToTrainImages = "VisDrone2019-DET-train/images"
pathToTrainAnnotations = "VisDrone2019-DET-train/annotations"
pathToValImages = "images"
pathToValAnnotatins = "annotations"

# define transforms
trainTransforms = get_transform(train=True)
valTransforms = get_transform(train=False)

trainDataset = VisDroneDataset(pathToTrainImages, pathToTrainAnnotations, transforms=trainTransforms)
valDataset = VisDroneDataset(pathToValImages, pathToValAnnotatins, transforms=valTransforms)

trainingLoader = DataLoader(trainDataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
valLoader = DataLoader(valDataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

# get the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", progress=True, num_classes=10)
num_classes = len(trainDataset.classes) # 11 classes (including background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

############ Define Hyperparameters #############
# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

############# Train Model #############
epochs = 2
for epoch in range(epochs):
    with model.no_grad():
        break
print("Training Finished!")

############ Test Model ##########


############ Get Results ###########


########## Convert Results into COCO #########