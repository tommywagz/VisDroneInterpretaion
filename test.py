import os
import json
import traceback
import numpy as np
from PIL import Image
from collections import deque, defaultdict
import datetime
import time
import sys

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class SmoothedValue(object):
    """Track a series of values and provide a smoothed version over a
    window of size `window_size`.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if sys.platform == "win32":
            log_msg = header + "[{0" + space_fmt + "}/{1}] eta: {eta} {meters}"
        else:
            log_msg = header + "[{0" + space_fmt + "}/{1}] eta: {eta} {meters}"
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            mem="{:.0f}M".format(torch.cuda.max_memory_allocated() / MB),
                        )
                    )
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")

class VisDroneDataset(Dataset):
    def __init__(self, root_dir, annotation_dir, transforms=None):
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted(os.listdir(root_dir))
        self.transforms = transforms
        self.annotations = self._load_annotations()
        self.classes = ['ignored region', 'pedestrian', 'people', 'car', 'van', 'bus', 'truck', 'tricycle', 'awning-tricycle', 'bicycle', 'motorcycle'] # VisDrone classes

    def _load_annotations(self):
        annotations = {}
        for img_name in self.image_files:
            annotation_name = img_name.replace('.jpg', '.txt')
            annotation_path = os.path.join(self.annotation_dir, annotation_name)
            boxes = []
            labels = []
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    for line in f:
                        try:
                            x, y, w, h, score, categoryID, truncation, occlusion = map(int, line.strip().split(',')[:8])
                            # VisDrone category IDs are 1-indexed (pedestrian to motorcycle)
                            if 1 <= categoryID <= 10:
                                boxes.append([x, y, x + w, y + h])
                                labels.append(categoryID)
                        except ValueError as e:
                            print(f"Error parsing line in {annotation_path}: {line.strip()} - {e}")
            annotations[img_name] = {'boxes': boxes, 'labels': labels}
        return annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        annotation_data = self.annotations[img_name]

        img = Image.open(img_path).convert("RGB")
        boxes = torch.as_tensor(annotation_data['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation_data['labels'], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # More augmentations can be added here
        # transforms.append(T.RandomPhotometricDistort())
        # transforms.append(T.RandomZoomOut())
        # transforms.append(T.RandomAdjustSharpness(0.3))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(len(data_loader) - 1, 100)

        def lr_lambda(step):
            if step < warmup_iters:
                alpha = float(step) / warmup_iters
                return warmup_factor * (1 - alpha) + alpha
            return 1

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

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
        outputs = model(images)

        outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator

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
        self.img_ids = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            if len(self.coco_eval) == 0:
                self.coco_eval[iou_type] = COCOeval(self.coco_gt, _create_coco_results(self.coco_gt, predictions, iou_type), iou_type)
            else:
                coco_dt = _create_coco_results(self.coco_gt, predictions, iou_type)
                self.coco_eval[iou_type].cocoDt = self.coco_eval[iou_type].cocoGt.loadRes(coco_dt)

    def synchronize_between_processes(self):
        # No explicit synchronization needed in a single-process setup
        pass

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

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
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], # Convert to [x, y, w, h]
                    "score": score,
                    "category_id": int(label),
                }
            )
        results.extend(coco_predictions)
    return results

def convertToCOCO(dataset):
    coco = COCO()
    coco.dataset = {
        "info": {"description": "VisDrone 2019 Dataset"},
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(dataset.classes[1:])], # Exclude 'ignored region'
        "images": [],
        "annotations": []
    }

    annotation_id = 1
    for i in range(len(dataset)):
        img_name = dataset.image_files[i]
        _, target = dataset[i]
        image_info = {"id": i, "file_name": img_name, "width": 0, "height": 0} # Actual width and height are not used in evaluation
        coco.dataset["images"].append(image_info)

        for j in range(len(target["boxes"])):
            bbox = target["boxes"][j].tolist()
            label = target["labels"][j].item()
            annotation = {
                "id": annotation_id,
                "image_id": i,
                "category_id": label,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], # Convert to [x, y, w, h]
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "iscrowd": 0
            }
            coco.dataset["annotations"].append(annotation)
            annotation_id += 1

    coco.createIndex()
    return coco

def main():
    # Define paths
    train_img_dir = "VisDrone2019-DET-train/images"
    train_anno_dir = "VisDrone2019-DET-train/annotations"
    val_img_dir = "images"
    val_anno_dir = "annotations"

    # Check if directories exist
    if not os.path.exists(train_img_dir) or not os.path.exists(train_anno_dir) or not os.path.exists(val_img_dir) or not os.path.exists(val_anno_dir):
        print("Error: One or more image or annotation directories not found. Please adjust the paths.")
        return

    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Define transforms
    train_transforms = get_transform(train=True)
    val_transforms = get_transform(train=False)

    # Load datasets
    train_dataset = VisDroneDataset(train_img_dir, train_anno_dir, transforms=train_transforms)
    val_dataset = VisDroneDataset(val_img_dir, val_anno_dir, transforms=val_transforms)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = len(train_dataset.classes) # 11 classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        # Train for one epoch
        metricLogger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
        lr_scheduler.step()

        # Evaluate on the validation set
        cocoEvaluator = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Validation AP: {cocoEvaluator.coco_eval['bbox'].stats[0]:.3f}")

        # Save checkpoint (optional)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, f'fasterrcnn_visdrone_epoch_{epoch}.pth')

    print("Training finished!")

if __name__ == "__main__":
    # Helper functions for distributed training (even if not using)
    import sys
    import torch.utils.data
    import utils

    def init_distributed():
        pass

    def cleanup_distributed():
        pass

    def get_world_size():
        return 1

    def is_main_process():
        return True

    def get_rank():
        return 0

    utils.init_distributed = init_distributed
    utils.cleanup_distributed = cleanup_distributed
    utils.get_world_size = get_world_size
    utils.is_main_process = is_main_process
    utils.get_rank = get_rank
    
    main()
