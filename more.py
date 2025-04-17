import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict, OrderedDict
import time
import math
import sys
import torch
import torch.amp as amp
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.models.detection as detection
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from test import MFasterRCNN, ModifiedFPN, ModifiedRoIHeads, ModifiedBoxHead
from utils import MetricLogger, SmoothedValue

# Define VisDroneDataset class 
class VisDroneDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None, num_classes=11):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted(os.listdir(img_dir))
        self.transforms = transforms
        # self.annotations = self._load_annotations()
        self.classes = ['ignored region', 'pedestrian', 'people', 'car', 'van', 'bus', 'truck', 'tricycle', 'awning-tricycle', 'bicycle', 'motorcycle'] # VisDrone classes
        self.num_classes = num_classes

    # def _load_annotations(self):
    #     annotations = {}
    #     for img_name in self.image_files:
    #         annotation_name = img_name.replace('.jpg', '.txt')
    #         annotation_path = os.path.join(self.annotation_dir, annotation_name)
    #         boxes = []
    #         labels = []
    #         if os.path.exists(annotation_path):
    #             with open(annotation_path, 'r') as f:
    #                 for line in f:
    #                     try:
    #                         x, y, w, h, score, categoryID, truncation, occlusion = map(int, line.strip().split(',')[:8])
    #                         if 1 <= categoryID <= 10:
    #                             x_min, y_min = x, y
    #                             x_max, y_max = x + w, y + h
    #                             if x_min < x_max and y_min < y_max:  # Check for valid bounding box
    #                                 boxes.append([x_min, y_min, x_max, y_max])
    #                                 labels.append(categoryID)
    #                             else:
    #                                 print(f"Warning: Invalid bounding box found and skipped in {annotation_path}: {line.strip()}")
    #                     except ValueError as e:
    #                         print(f"Error parsing line in {annotation_path}: {line.strip()} - {e}")
    #         annotations[img_name] = {'boxes': boxes, 'labels': labels}
    #     return annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        anno_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))  # Adjust extension as needed

        # annotation_data = self.annotations[img_name]

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []

        try:
            with open(anno_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        parts = list(map(int, line.split(',')))
                    except ValueError:
                        continue
                    if len(parts) >= 8:
                        x1, y1, w, h, _, label, _, _ = parts[:8]
                        if w > 0 and h > 0 and label > 0 and label < self.num_classes:  # Basic check for valid bounding box and label
                            x2 = x1 + w
                            y2 = y1 + h
                            boxes.append([x1, y1, x2, y2])
                            labels.append(label)
                        # else:
                        #     print(f"Warning: Invalid bounding box found and skipped in {anno_path}: {line.strip()}")


        except FileNotFoundError:
            print(f"Annotation file not found: {anno_path}")
            pass # Return empty lists

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.zeros(0) #added check
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes.numel() > 0 else torch.zeros(0) #added check

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            boxes = target['boxes']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.tolist()

            labels = [int(label) for label in target['labels']]
            
            transformed = self.transforms(
                image=np.array(img),
                bboxes=boxes,
                labels=labels
            )
            
            img = F.to_tensor(transformed['image'])
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        assert torch.all((target['labels'] >= 0) & (target['labels'] < self.num_classes)), f"Bad labels: {target['labels']}"

        return img, target

def get_transform(train):
    transforms = []
    if train:
        transforms.extend([
            # Add more small object focused augmentations here
            # A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),
            A.Resize(height=320, width=320)
            # A.RandomCrop(width=512, height=512, p=0.5),
            # A.HorizontalFlip(p=0.5),
        ])
    else:
        transforms.append(A.Resize(height=320, width=320))
        transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        
    # transforms.extend([
    #     A.ToFloat(max_value=255.0),
    #     ToTensorV2()
    # ])
    
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

def collate_fn(batch):
    return tuple(zip(*batch))       

def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = convertToCOCO(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for (images, targets) in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        with amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            outputs = model(images)

        outputs = [{k: v.to(torch.device('cpu')) for k , v in t.items()} for t in outputs]

        results = []
        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()
            if 'boxes' in output:
                boxes = output['boxes'].cpu().tolist()
                scores = output['scores'].cpu().tolist()
                labels = output['labels'].cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        'image_id': image_id,
                        'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        'score': score,
                        'category_id': label
                    })
            
        coco_evaluator.update(results)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(len(data_loader) - 1, 100)

        def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
            def f(x):
                if x >= warmup_iters:
                    return 1
                alpha = float(x) / warmup_iters
                return warmup_factor * (1 - alpha) + alpha

            return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs if needed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            losses_reduced = reduce_dict(loss_dict)
            losses = sum(loss for loss in losses_reduced.values())
            loss_dict_reduced = {
                k: v.item() for k, v in losses_reduced.items()
            }
        else:
            loss_dict_reduced = {
                k: v.item() for k, v in loss_dict.items()
            }

        if not math.isfinite(losses):
            print("Loss is {}, stopping training".format(losses))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def reduce_dict(input_dict, average=True):
    """
    Reduces (sum) the values of a dict from all processes in the world.
    Optionaly takes the average of the values over all processes.
    Args:
        input_dict (dict): a dictionary of tensors to be reduced.
        average (bool): whether to take the average of the outcomes.
            Default: True
    Returns:
        a dict with the same fields as input_dict, with the values
        being the sum of all values
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so the values are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values,dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict

def get_world_size():
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

# COCO Evaluator and convertToCOCO functions (same as before)
class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"iou_types must be a list or tuple of strings, got {iou_types}")
        allowed_iou_types = ("bbox", "segm")
        for iou_type in iou_types:
            if iou_type not in allowed_iou_types:
                raise ValueError(f"iou_type: {iou_type} not in {allowed_iou_types}")
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        self.predictions = defaultdict(list)

    def update(self, predictions):
        for image_id, prediction in predictions.items():
            self.predictions[image_id].append(prediction)
            
    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        for iou_type in self.iou_types:
            coco_dt = _create_coco_results(self.coco_gt, self.predictions, iou_type)
            coco_dt_coco = COCO()
            coco_dt_coco.dataset = {'images': self.coco_gt.dataset['images'], 'annotations': coco_dt, 'categories': self.coco_gt.dataset['categories']}
            coco_dt_coco.createIndex()

            coco_eval = COCOeval(self.coco_gt, coco_dt_coco, iou_type)
            coco_eval.params.imgIds = list(self.predictions.keys()) # Evaluate on the images with predictions
            coco_eval.evaluate()
            coco_eval.accumulate()
            self.coco_eval[iou_type] = coco_eval
            
    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    @property
    def results(self):
        return {iou_type: coco_eval.stats.tolist() for iou_type, coco_eval in self.coco_eval.items()}

def _create_coco_results(coco_gt, predictions, iou_type):
    results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"].tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_predictions = []
        for box, score, label in zip(boxes, scores, labels):
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": score,
                    "category_id": int(label),
                }
            )
        results.extend(coco_predictions)
    return results

def convertToCOCO(dataset):
    coco_dict = {}
    coco_dict['images'] = []
    coco_dict['annotations'] = []
    coco_dict['categories'] = []

    image_id_counter = 0
    annotation_id_counter = 0
    category_id_map = {}
    current_category_id = 1
    
    if hasattr(dataset, 'classes'):
        for i, class_name in enumerate(dataset.classes):
            coco_dict['categories'].append({'id': i + 1, 'name': class_name})
            category_id_map[class_name] = i + 1
    else:
        # Handle the case where dataset.classes is not available
        # infer categories from the labels in the annotations
        unique_labels = set()
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            if 'labels' in target:
                for label in target['labels'].tolist():
                    unique_labels.add(label)

        sorted_labels = sorted(list(unique_labels))
        for i, label in enumerate(sorted_labels):
            coco_dict['categories'].append({'id': i + 1, 'name': str(label)})
            category_id_map[label] = i + 1

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        img_height, img_width = img.shape[-2:]
        image_id = image_id_counter
        coco_dict['images'].append({'id': image_id, 'width': img_width, 'height': img_height})
        image_id_counter += 1

        if 'boxes' in target:
            for i in range(len(target['boxes'])):
                bbox = target['boxes'][i].tolist()
                label = target['labels'][i].item()

                # Get category_id from the map (using class name if available, else label index)
                category_id = -1
                if hasattr(dataset, 'classes'):
                    class_name = dataset.classes[label]
                    category_id = category_id_map.get(class_name)
                else:
                    category_id = category_id_map.get(label)

                if category_id is not None:
                    area = bbox[2] * bbox[3]  # Calculate area
                    coco_dict['annotations'].append({
                        'id': annotation_id_counter,
                        'image_id': image_id,
                        'bbox': bbox,
                        'area': area,
                        'category_id': category_id,
                        'iscrowd': 0,  # Assuming not a crowd
                    })
                    annotation_id_counter += 1

    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco

def generate_predictions(model, data_loader, device):
    model.eval()
    results = []
    for idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]['image_id'].item()
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                results.append({
                    'image_id': image_id,
                    'bbox': box.tolist(),
                    'score': score.item(),
                    'category_id': label.item(),
                })
                
    return results


def main():
    # Define paths
    train_img_dir = "VisDrone2019-DET-train/images"
    train_anno_dir = "VisDrone2019-DET-train/annotations"
    val_img_dir = "images"
    val_anno_dir = "annotations"
    # test_img_dir = "VisDrone2019-DET-test-dev/images" # Adjust if you have ground truth for test
    # test_anno_dir = "VisDrone2019-DET-test-dev/annotations" # Adjust if you have ground truth for test

    # Check if directories exist
    if not os.path.exists(train_img_dir) or not os.path.exists(train_anno_dir) or not os.path.exists(val_img_dir) or not os.path.exists(val_anno_dir):
        print("Error: One or more image or annotation directories not found. Please adjust the paths.")
        return
    # if not os.path.exists(test_img_dir):
    #     print("Warning: Test image directory not found.")

    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Define transforms
    train_transforms = get_transform(train=True)
    val_transforms = get_transform(train=False)
    # test_transforms = get_transform(train=False) # Usually no strong augmentation for test

    # Load datasets
    train_dataset = VisDroneDataset(train_img_dir, train_anno_dir, transforms=train_transforms)
    val_dataset = VisDroneDataset(val_img_dir, val_anno_dir, transforms=val_transforms)
    # test_dataset = VisDroneDataset(test_img_dir, test_anno_dir, transforms=test_transforms) # If test annotations are available

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn) # If test annotations are available

    # Define hyperparameters (reflecting the paper)
    num_classes = 11
    learning_rate = 0.0001 # Assuming a tuned learning rate
    weight_decay = 0.0005 # Assuming a tuned weight decay
    num_epochs = 5
    print_freq = 100
    lr_step_size = 3
    lr_gamma = 0.1

    # Load pre-trained ResNet-34 backbone
    mobilenet_backbone = models.mobilenet_v3_large(weights='DEFAULT').features
    
    # Downsampling to improve efficiency    
    extra_layers = nn.Sequential(
        nn.Conv2d(960, 512, kernel_size=3, stride=2, padding=1),  # 10x10 -> 5x5
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # 5x5 -> 3x3
        nn.ReLU(inplace=True),
    )
    
    # Define the layers we want to use as feature maps for SSD
    # You'll need to inspect the MobileNetV3 architecture to determine suitable layers
    # These layer indices might need adjustment based on the exact architecture
    feature_layers = OrderedDict()
    feature_layers['0'] = mobilenet_backbone[0:2]  # Example: First few layers
    feature_layers['1'] = mobilenet_backbone[2:4]  # Example: Next few layers
    feature_layers['2'] = mobilenet_backbone[4:7]
    feature_layers['3'] = mobilenet_backbone[7:10]
    feature_layers['4'] = mobilenet_backbone[10:13]
    feature_layers['5'] = mobilenet_backbone[13:]
    feature_layers['6'] = extra_layers[0:2]
    feature_layers['7'] = extra_layers[2:]   
            
    # Define the output channels for each feature map
    # You'll need to get these from the MobileNetV3 architecture definition
    feature_channels = [16, 24, 40, 80, 112, 960, 512, 256] 
    
    # Create the backbone using the defined feature layers
    backbone = torch.nn.Sequential(feature_layers)
    backbone.out_channels = feature_channels # Set the out_channels attribute
    
    # Define anchor generator with adjusted parameters
    anchor_generator = DefaultBoxGenerator(
        min_ratio=20,
        max_ratio=90,
        steps=[8, 16, 32, 64, 128, 256, 300, 300],
        aspect_ratios=([2, 3],) * len(feature_channels)
    )
    
    # Define the SSD head with updated channel info
    num_anchors = anchor_generator.num_anchors_per_location()
    
    # create SSD head with correct input channels and anchor sizes
    ssd_head = ssd.SSDHead(
        in_channels=feature_channels[::-1],
        num_anchors=num_anchors[::-1],
        num_classes=num_classes
    )

    # Create your custom model
    model = torchvision.models.detection.ssd300_vgg16(num_classes=11)
    
    model.to(device)

    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    scaler = amp.GradScaler() if torch.cuda.is_available() else None

    # Training loop  potentially with the ground truth box limit
    for epoch in range(num_epochs):
        # Modify train_one_epoch if you need to limit ground truth boxes
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq, scaler)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Validation AP: {coco_evaluator.coco_eval['bbox'].stats[0]:.3f}")
        # Save checkpoint

    print("Training finished!")

    # Evaluation on test set 

if __name__ == '__main__':
    main()

