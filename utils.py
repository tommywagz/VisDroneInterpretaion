import datetime
import time
import sys
import math
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import ssd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F


class SSDMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mobilenet_backbone = models.mobilenet_v3_large(weights='DEFAULT').features

        print(self.mobilenet_backbone)

        feature_layers_indices = [0, 2, 4, 7, 13, 16]
        self.feature_extractor = nn.ModuleList([self.mobilenet_backbone[i] for i in feature_layers_indices])

        feature_channels = [16, 24, 40, 80, 160, 960]

        # Extra layers (as defined before)
        self.extra_layers = nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.extra_feature_layers = nn.ModuleList([self.extra_layers[i] for i in range(len(self.extra_layers)) if i % 2 == 0])
        feature_channels.extend([512, 256]) # Add output channels of extra layers

        anchor_generator = ssd.DefaultBoxGenerator(
            min_ratio=3,
            max_ratio=30,
            steps=[8, 16, 32, 64, 128, 320//10, 320//5, 320//20], # Adjust steps based on feature map sizes
            aspect_ratios=([0.5, 1.0, 2.0],) * len(feature_channels)
        )
        num_anchors = anchor_generator.num_anchors_per_location()
        self.ssd_head = ssd.SSDHead(feature_channels, num_anchors, num_classes)

    def _get_out_channels(self, layer):
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels
        elif isinstance(layer, nn.Sequential):
            for module in reversed(layer):
                if isinstance(module, nn.Conv2d):
                    return module.out_channels
        return None

    def forward(self, images):
        features = []
        x = torch.stack(images)
        for layer in self.feature_extractor:
            x = layer(x)
            features.append(x)

        for layer in self.extra_feature_layers:
            x = layer(x)
            features.append(x)

        return self.ssd_head(features) 
    
    
        # # Define anchor generator with adjusted parameters
    # anchor_generator = DefaultBoxGenerator(
    #     min_ratio=3,
    #     max_ratio=30,
    #     steps=[8, 16, 32, 64, 128, 320//10, 320//5, 320//3],
    #     aspect_ratios=([0.5, 1.0, 2.0],) * 8
    # )
    
    # # Define the SSD head with updated channel info
    # num_anchors = anchor_generator.num_anchors_per_location()

    # # Load pre-trained mobilenet-34 backbone
    # mobilenet_backbone = models.mobilenet_v3_large(weights='DEFAULT').features
    # print("Feature layer output channels")
    # print(mobilenet_backbone)

    
    # # Define the layers we want to use as feature maps for SSD
    # feature_layers = OrderedDict()
    # feature_layers['0'] = mobilenet_backbone[2].block  
    # feature_layers['1'] = mobilenet_backbone[4].block 
    # feature_layers['2'] = mobilenet_backbone[7].block
    # feature_layers['3'] = mobilenet_backbone[13].block
    # feature_layers['4'] = mobilenet_backbone[16]
    
    # # Downsampling to improve efficiency    
    # extra_layers = nn.Sequential(
    #     nn.Conv2d(960, 512, kernel_size=3, stride=2, padding=1),  # 10x10 -> 5x5
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # 5x5 -> 3x3
    #     nn.ReLU(inplace=True),
    # )
    
    # feature_layers['5'] = extra_layers[0:2]
    # feature_layers['6'] = extra_layers[2:4]   
            
    # # Define the output channels for each feature map
    # # You'll need to get these from the MobileNetV3 architecture definition
    # feature_channels = [16, 24, 40, 80, 160, 960, 512, 256] 
                
    # # Create the backbone using the defined feature layers
    # backbone = torch.nn.Sequential(feature_layers)
    # backbone.out_channels = list(feature_channels) # Set the out_channels attribute
    
    # # create SSD head with correct input channels and anchor sizes
    # ssd_head = ssd.SSDHead(
    #     in_channels=list(feature_channels),
    #     num_anchors=num_anchors,
    #     num_classes=num_classes
    # )  

# Modify the train_one_epoch function to calculate the loss
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()

    # Freeze BatchNorm layers in the MobileNetV3 backbone
    # for m in model.mobilenet_backbone.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eval()

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

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        if batch is None:
            continue

        images, targets = batch
        # print(f"Image data type: {images[0].dtype}")
        # print(f"Image min: {images[0].min()}, Image max: {images[0].max()}")
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            # output = model(images) # Get predictions from the model
            loss_dict = model(images, targets) # Assuming your SSDHead has a loss method
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
            print(f"Loss is {losses}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
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

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item() if len(self.deque) > 0 else 0.0

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() if len(self.deque) > 0 else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t"):
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
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.value, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        max_memory = 0
        iter_time = SmoothedValue(fmt='{global_avg:.4f}')
        data_time = SmoothedValue(fmt='{global_avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            i += 1
            if i % print_freq == 0 or i == len(iterable):
                batch_time = time.time() - end
                iter_time.update(batch_time)
                if torch.cuda.is_available():
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=max_memory
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def is_dist_avail_and_initialized():
    # Simple placeholder function
    return False

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

# Helper function to draw boxes
def draw_boxes(ax, boxes, labels, color, label_map, scores=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label_str = label_map.get(labels[i].item(), str(labels[i].item()))
        if scores is not None:
            label_str += f' ({scores[i]:.2f})'
        ax.text(x1, y1 - 5, label_str, color=color, fontsize=8, backgroundcolor='white')

# Main function
def visualize_sample(model, dataset, idx=0, device='cuda', label_map=None, score_thresh=0.3):
    model.eval()
    img, target = dataset[idx]
    
    # Move to device
    input_tensor = img.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)[0]

    # Filter low scores
    keep = output['scores'] >= score_thresh
    pred_boxes = output['boxes'][keep]
    pred_labels = output['labels'][keep]
    pred_scores = output['scores'][keep]

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(F.to_pil_image(img))

    draw_boxes(ax, target['boxes'], target['labels'], 'green', label_map)
    draw_boxes(ax, pred_boxes, pred_labels, 'red', label_map, pred_scores)

    plt.title(f"Green: GT, Red: Prediction (idx={idx})")
    plt.axis('off')
    plt.show()