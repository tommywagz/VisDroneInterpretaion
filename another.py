import albumentations as A
from albumentations.pytorch import ToTensorV2

class FilterInvalidBBoxesAndLabels(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(FilterInvalidBBoxesAndLabels, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img

    def update_params(self, params, **kwargs):
        return params

    def apply_to_bboxes(self, bboxes, **params):
        labels = params["labels"]
        valid_bboxes = []
        valid_labels = []
        for bbox, label in zip(bboxes, labels):
            if bbox[2] > 1 and bbox[3] > 1:  # width > 1, height > 1
                valid_bboxes.append(bbox)
                valid_labels.append(label)
        params["labels"][:] = valid_labels  # Update in-place
        return valid_bboxes


def get_transform(train=True):
    transforms = [
        A.Resize(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # normalize before tensor
        ToTensorV2()
    ]

    if train:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.3),
            A.HueSaturationValue(p=0.3),
            FilterInvalidBBoxesAndLabels(),
        ] + transforms  # add normalization and tensor conversion last

    else:
        transforms = [
            FilterInvalidBBoxesAndLabels(),
        ] + transforms

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
    )








# class FixInvalidBBoxes(A.ImageOnlyTransform):
#     """
#     Custom transform to aggressively fix invalid bounding boxes.
#     Explicitly enforces a small positive width and height after clipping.
#     """
#     def __init__(self, min_area=1, always_apply=False, p=1.0):
#         super().__init__(always_apply=always_apply, p=p)
#         self.min_area = min_area

#     def apply_to_bboxes(self, bboxes, **params):
#         fixed_bboxes = []
#         for bbox in bboxes:
#             x_min, y_min, w, h = bbox[:4]
#             x_max = x_min + w
#             y_max = y_min + h

#             clipped_x_min = max(0.0, min(1.0, x_min))
#             clipped_y_min = max(0.0, min(1.0, y_min))
#             clipped_x_max = max(0.0, min(1.0, x_max))
#             clipped_y_max = max(0.0, min(1.0, y_max))

#             clipped_w = clipped_x_max - clipped_x_min
#             clipped_h = clipped_y_max - clipped_y_min

#             # Enforce a minimum positive width and height
#             if clipped_w <= 0:
#                 clipped_w = 1e-6
#                 clipped_x_max = clipped_x_min + 1e-6
#             if clipped_h <= 0:
#                 clipped_h = 1e-6
#                 clipped_y_max = clipped_y_min + 1e-6

#             if clipped_w * clipped_h >= self.min_area:
#                 fixed_bboxes.append([clipped_x_min, clipped_y_min, clipped_w, clipped_h] + list(bbox[4:]))
#         return fixed_bboxes

#     def get_transform_init_args_names(self):
#         return ("min_area", "always_apply", "p")