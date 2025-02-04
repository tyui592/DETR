# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as T
import numpy as np
import albumentations as A
from utils.box_ops import box_xyxy_to_cxcywh


def get_transforms(args):
    transforms = Albumentation(args.aug_policy, args.image_size, args.image_set)
    collate_func = collate_fn
    return transforms, collate_func


class Albumentation:
    def __init__(self, aug_policy, target_size, image_set):
        self.transforms = get_shwd_transforms(policy=aug_policy,
                                              target_size=target_size,
                                              image_set=image_set)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, img, target):
        img_np = np.asarray(img)
        mask = np.ones_like(img_np)[:, :, 0]

        new = self.transforms(image=img_np,
                              bboxes=target['bboxes'],
                              labels=target['labels'],
                              mask=mask)

        h, w = new['image'].shape[:2]
        bboxes = torch.tensor(new['bboxes'])
        if bboxes.shape[0] > 0:
            box_scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            bboxes = box_xyxy_to_cxcywh(bboxes) / box_scale

        item = {
            'image': self.normalize(new['image']),
            'mask': torch.tensor(new['mask'], dtype=torch.float32),
            'bboxes': bboxes,
            'size': torch.tensor([h, w]),
            'labels': torch.tensor(new['labels'], dtype=torch.long)
        }
        return item

def collate_fn(batch):
    image_lst = [item['image'] for item in batch]
    mask_lst = [item['mask'] for item in batch]
    bboxes_lst = [item['bboxes'] for item in batch]
    labels_lst = [item['labels'] for item in batch]
    file_id_lst = [item['file_id'] for item in batch]
    raw_size_lst = [item['raw_size'] for item in batch]
    size_lst = [item['size'] for item in batch]

    inputs = {
        'images': torch.stack(image_lst, dim=0),
        'masks': torch.stack(mask_lst, dim=0),
        'file_ids': file_id_lst,
        'raw_sizes': torch.tensor(raw_size_lst),
        'sizes': torch.stack(size_lst)
    }

    targets = [
        {
            'labels': labels, 'boxes': bboxes
        } for labels, bboxes in zip(labels_lst, bboxes_lst)
    ]

    return inputs, targets

def get_shwd_transforms(policy, target_size=608, image_set='trainval'):
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])
    if image_set == 'test':
        transforms = A.Compose([
            A.Resize(height=target_size, width=target_size, p=1.0)
        ], bbox_params=bbox_params)
    else:
        if policy == 1:
            transforms = A.Compose([
                A.SmallestMaxSize(max_size=target_size, p=1.0),
                A.RandomSizedBBoxSafeCrop(height=target_size,
                                          width=target_size,
                                          erosion_rate=0.2, p=1.0),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=bbox_params)
        elif policy == 2:
            transforms = A.Compose([
                A.SmallestMaxSize(max_size=target_size, p=1.0),
                A.Affine(scale=(0.5, 1.5), translate_percent=0.1, p=0.5),
                A.RandomSizedBBoxSafeCrop(height=target_size,
                                          width=target_size,
                                          erosion_rate=0.2, p=1.0),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=bbox_params)
        elif policy == 3:
            transforms = A.Compose([
                A.Affine(scale=(0.5, 1.5), translate_percent=0.1, p=1.0),
                A.BBoxSafeRandomCrop(erosion_rate=0.1, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=target_size, width=target_size, p=1.0),
                A.ColorJitter(p=0.5),
            ], bbox_params=bbox_params)
        elif policy == 4:
            transforms = A.Compose([
                A.Affine(scale=(0.8, 1.2), translate_percent=0.05, p=1.0),
                A.BBoxSafeRandomCrop(erosion_rate=0.05, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=target_size, width=target_size, p=1.0),
                A.ColorJitter(p=0.5),
            ], bbox_params=bbox_params)
        elif policy == 5:  # can make no-object image
            transforms = A.Compose([
                A.Affine(scale=(0.8, 1.2), translate_percent=0.05, p=1.0),
                A.BBoxSafeRandomCrop(erosion_rate=0.05, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=target_size, width=target_size, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            ], bbox_params=bbox_params)
    return transforms
