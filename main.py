import os
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict, OrderedDict

import torch, torchvision
import torch.amp as amp
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import torchvision.transforms.functional as F

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import MetricLogger, SSDMobileNetV3, visualize_sample, train_one_epoch
from another import get_transform

# Define VisDroneDataset class 
class VisDroneDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None, num_classes=12):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted([
            f.name for f in Path(self.img_dir).iterdir()
            if f.is_file() and f.suffix.lower() == '.jpg'
        ])        
        self.transforms = transforms
        self.classes = ['ignored region', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'] # VisDrone classes
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        anno_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.txt'))
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        boxes = []
        labels = []

        try:
            with open(anno_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        x, y, w, h, score, label, truncation, occlusion = map(float, parts[:8])

                        if w > 0 and h > 0 and 0 <= x < width and 0 <= y < height and x + w <= width and y + h <= height:
                            if 0 < label < self.num_classes and score > 0:
                                boxes.append([x, y, w, h])
                                labels.append(label)


                    except ValueError:
                        print(f"Annotation '{line}' at file '{anno_path}' could not be read")
                        continue

        except FileNotFoundError:
            print(f"Annotation file not found: {anno_path}")
            pass # Return empty lists

        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (torch.as_tensor(boxes, dtype=torch.float32)[:, 2] * torch.as_tensor(boxes, dtype=torch.float32)[:, 3]) if boxes else torch.zeros(0)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64) if boxes else torch.zeros(0)

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['origin_size'] = torch.as_tensor([int(height), int(width)])
        target['size'] = torch.as_tensor([int(height), int(width)])

        if self.transforms is not None:
            # bboxes_to_transform = target['boxes'].tolist() if target['boxes'].numel() > 0 else []
            # labels_to_transform = labels.tolist() if labels.numel() > 0 else []

            transformed = self.transforms(
                image=np.array(img),
                bboxes=target['boxes'].tolist(),
                labels=target['labels'].tolist()
            )

            img = transformed['image']
                
            # target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            # target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

            # target["boxes"][:, 2] = target["boxes"][:, 0] + target["boxes"][:, 2]
            # target["boxes"][:, 3] = target["boxes"][:, 1] + target["boxes"][:, 3]

            if len(transformed["bboxes"]) == 0:
                print(f"WARNING: Empty boxes for image {img_path}")
                # print(f"Original target: {target}")
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['labels'] = torch.empty((0,), dtype=torch.int64)
            else:
                target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

                # Convert COCO -> Pascal VOC
                if target["boxes"].ndim == 2 and target["boxes"].size(0) > 0:
                    target["boxes"][:, 2] = target["boxes"][:, 0] + target["boxes"][:, 2]
                    target["boxes"][:, 3] = target["boxes"][:, 1] + target["boxes"][:, 3]
                
            # if len(target["boxes"]) > 0:
            #     print(f"Sample box: {target['boxes'][0]}")
        
        assert target['boxes'].dim() == 2 and target['boxes'].size(1) == 4, f"Invalid box shape: {target['boxes'].shape}"

        return img, target


def collate_fn(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None  # or raise an error, depending on your strategy

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
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        with amp.autocast(device_type=device.type, dtype=dtype):
            outputs = model(images)


        outputs = [{k: v.to(torch.device('cpu')) for k , v in t.items()} for t in outputs]

        results = []
        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()
            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu()

            for box, score, label in zip(boxes, scores, labels):
                bbox = [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()]
                if box[2] > 0 and box[3] > 0:
                    results.append({
                        'image_id': image_id,
                        'bbox': bbox,
                        'score': score.item(),
                        'category_id': label.item()
                    })

        coco_evaluator.update(results)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


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
        for prediction in predictions:
            image_id = prediction["image_id"]
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
    id = 1
    for image_id, prediction in predictions.items():
        for pred in prediction:
            x, y, w, h = pred['bbox']
            pred['area'] = w * h
            pred['id'] = id
            id += 1
            results.append(pred)
            
    return results

def convertToCOCO(dataset):
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    category_id_map = {}
    annotation_id_counter = 0

    # Define category mapping
    if hasattr(dataset, 'classes'):
        for i, class_name in enumerate(dataset.classes):
            category_id = i + 1  # COCO category_ids must be >= 1
            coco_dict['categories'].append({'id': category_id, 'name': class_name})
            category_id_map[i] = category_id  # map original label index to COCO category_id
    else:
        # Fallback: infer from labels directly
        unique_labels = set()
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            if 'labels' in target:
                unique_labels.update(target['labels'].tolist())

        for i, label in enumerate(sorted(unique_labels)):
            category_id = i + 1
            coco_dict['categories'].append({'id': category_id, 'name': str(label)})
            category_id_map[label] = category_id

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        img_height, img_width = img.shape[-2:]

        # image_id should match what will be used in predictions
        image_id = idx
        coco_dict['images'].append({
            'id': image_id,
            'width': img_width,
            'height': img_height
        })

        if 'boxes' in target and 'labels' in target:
            boxes = target['boxes']
            labels = target['labels']

            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].tolist()
                w = max(x_max - x_min, 1e-2)  # Avoid 0-area boxes
                h = max(y_max - y_min, 1e-2)
                bbox = [x_min, y_min, w, h]

                label = labels[i].item()
                category_id = category_id_map.get(label)

                if category_id is None:
                    continue  # Skip unknown labels

                coco_dict['annotations'].append({
                    'id': annotation_id_counter,
                    'image_id': image_id,
                    'bbox': bbox,
                    'area': w * h,
                    'category_id': category_id,
                    'iscrowd': 0,
                })
                annotation_id_counter += 1

    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


# def generate_predictions(model, data_loader, device):
#     model.eval()
#     results = []
#     for idx, (images, targets) in enumerate(data_loader):
#         images = list(image.to(device) for image in images)
#         with torch.no_grad():
#             outputs = model(images)

#         for i, output in enumerate(outputs):
#             image_id = targets[i]['image_id'].item()
#             boxes = output['boxes'].cpu().numpy()
#             scores = output['scores'].cpu().numpy()
#             labels = output['labels'].cpu().numpy()

#             for box, score, label in zip(boxes, scores, labels):
#                 results.append({
#                     'image_id': image_id,
#                     'bbox': box.tolist(),
#                     'score': score.item(),
#                     'category_id': label.item(),
#                 })
                
#     return results


def main():
    # Define the main training data directories
    train_img_dir = "VisDrone2019-DET-train/images"
    train_anno_dir = "VisDrone2019-DET-train/annotations"
    val_img_dir = "VisDrone2019-DET-val/images"
    val_anno_dir = "VisDrone2019-DET-val/annotations"
    test_img_dir = "VisDrone2019-DET-test/images"
    test_anno_dir = "VisDrone2019-DET-test/annotations"

    # Get the list of image files (assuming annotations have the same base name)
    # image_files = sorted(os.listdir(train_img_dir))
    # annotation_files = [f.replace('.jpg', '.txt') for f in image_files]

    # Check if directories exist
    if not os.path.exists(train_img_dir) or not os.path.exists(train_anno_dir) or not os.path.exists(val_img_dir) or not os.path.exists(val_anno_dir) or not os.path.exists(test_img_dir) or not os.path.exists(test_anno_dir):
        print("Error: One or more image or annotation directories not found. Please adjust the paths.")
        return
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(f"Using {device}")

    # print(f"Torchvision version: {torchvision.__version__}")
    # print(f"Albumentation version: {A.__version__}")

    # Define transforms
    train_transforms = get_transform(train=True)
    val_transforms = get_transform(train=False)

    # Load datasets image_files=train_img_files, annotation_files=train_anno_files, image_files=val_img_files, annotation_files=val_anno_files
    num_classes = 12
    train_dataset = VisDroneDataset(train_img_dir, train_anno_dir, transforms=train_transforms, num_classes=num_classes)
    val_dataset = VisDroneDataset(val_img_dir, val_anno_dir,  transforms=val_transforms, num_classes=num_classes)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True) # Set your desired batch size and num_workers
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn, drop_last=False)   # Set your desired batch size and num_workers

    # Define hyperparameters (reflecting the paper)
    learning_rate = 0.0005 
    weight_decay = 0.0005 
    num_epochs = 50
    print_freq = 100
    lr_step_size = 30
    lr_gamma = 0.1

    # Create model
    # model = SSDMobileNetV3(num_classes=num_classes)
    # model = ssd.SSD(backbone=backbone, anchor_generator=anchor_generator, size=(30,30), num_classes=num_classes, head=ssd_head)
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    # model.transform.min_size = (512,)
    # model.transform.max_size = 1024
    # model = ssd.ssd300_vgg16(
    #     weights=ssd.SSD300_VGG16_Weights.DEFAULT 
    #     # progress=True, 
    #     # num_classes=num_classes, 
    #     # weights_backbone=torchvision.models.VGG16_Weights, 
    #     # trainable_backbone_layers=5
    # )
    model.to(device)

    # Optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    scaler = amp.GradScaler() if torch.cuda.is_available() else None

    # Training loop  potentially with the ground truth box limit
    for epoch in range(num_epochs):
        # Modify train_one_epoch if you need to limit ground truth boxes
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq, scaler)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Validation AP: {coco_evaluator.coco_eval['bbox'].stats[0]:.3f}")
        if epoch == num_epochs - 1 or epoch == 1:
            label_map = {i: name for i, name in enumerate(val_loader.dataset.classes)}
            visualize_sample(model, val_loader.dataset, idx=5, device=device, label_map=label_map)

        # Save checkpoint

    print("Training finished!")

    # Evaluation on test set 

if __name__ == '__main__':
    main()

