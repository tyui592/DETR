# -*- coding: utf-8 -*-
"""Data Code for SafetyHelmetWearing-Dataset(SHWD).

- Ref: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
"""

import torch
from PIL import Image
from utils.misc import parse_xml
from datasets.transforms import get_transforms


def get_shwd_dataloader(args, shuffle=True, drop_last=True):
    transforms, collate_fn = get_transforms(args)

    dataset = SHWDataset(data_root=args.data_root,
                         data_set=args.image_set,
                         transforms=transforms)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=shuffle,
                                             num_workers=args.num_workers,
                                             drop_last=drop_last,
                                             collate_fn=collate_fn,
                                             pin_memory=args.pin_memory,
                                             pin_memory_device=args.device if args.pin_memory else '')

    return dataloader


class SHWDataset:
    def __init__(self,
                 data_root,
                 transforms,
                 data_set='trainval'):
        self.image_root = data_root / 'JPEGImages/'
        self.label_root = data_root / 'Annotations/'
        self.transforms = transforms

        set_path = data_root / 'ImageSets' / 'Main' / f'{data_set}.txt'
        self.data_lst = [line for line in set_path.read_text().split('\n') if line]

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, index):
        file_id = self.data_lst[index]
        image_path = self.image_root / f'{file_id}.jpg'
        label_path = self.label_root / f'{file_id}.xml'

        img = Image.open(image_path).convert('RGB')
        bboxes, labels = parse_xml(label_path)
        target = {'bboxes': bboxes, 'labels': labels, 'file_id': file_id}

        item = self.transforms(img, target)

        item['file_id'] = file_id
        w, h = img.size
        item['raw_size'] = h, w

        return item
