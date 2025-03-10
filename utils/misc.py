# -*- coding: utf-8 -*-
"""Utility Code."""

import logging
import pickle
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from collections import defaultdict

from utils.box_ops import box_cxcywh_to_xyxy


def draw_box(pil, box, width=2, color=(0, 0, 255)):
    """Draw a box(xyxy) on a pil."""
    draw = ImageDraw.Draw(pil)
    draw.rectangle(list(map(int, box)), width=width, outline=color, fill=None)
    return pil


def count_parameters(model):
    """Count number of model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """AverageMeter."""

    def __init__(self):
        """Set zero value."""
        self.reset()

    def reset(self):
        """Reset."""
        self.avg = 0
        self.sum = 0
        self.count = 0

        return None

    def update(self, val, n=1):
        """Update value."""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        return None


def save_dict(path, dic):
    """Save a dict."""
    with open(path, 'wb') as f:
        pickle.dump(dic, f)
    return


def load_dict(path):
    """Load a dict."""
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def get_logger(path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt=f"[%(asctime)s | {name}]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file hander
    file_handler = logging.FileHandler(path / 'info.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


@torch.no_grad()
def draw_outputs(inputs, targets, results, num_image=2):
    mean = [-m / s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    std = [1 / s for s in (0.229, 0.224, 0.225)]
    pils = []
    colors = {0: 'red', 1: 'blue', 2: 'black'}
    colors_gt = {0: 'green', 1: 'yellow'}
    imgs = TF.normalize(inputs['images'], mean=mean, std=std)
    for i in range(num_image):
        pil = TF.to_pil_image(imgs[i].cpu())
        
        # draw predictions
        for box, label in zip(results[i]['boxes'].tolist(),
                              results[i]['labels'].tolist()):
            pil = draw_box(pil, box, color=colors[label])

        # draw gt boxes
        if len(targets[i]['boxes']) != 0:
            h, w = inputs['sizes'][i]
            box_scale = torch.tensor([[w, h, w, h]])
            gt_boxes = box_cxcywh_to_xyxy(targets[i]['boxes'].cpu()) * box_scale.cpu()
            for box, label in zip(gt_boxes, targets[i]['labels'].tolist()):
                pil = draw_box(pil, box, color=colors_gt[label])
        pils.append(pil)

    return pil_concat_h(pils)


def pil_concat_h(pils):
    width, height = 0, 0
    for pil in pils:
        width += pil.width
        height = pil.height
    new = Image.new('RGB', (width, height))
    s = 0
    for pil in pils:
        new.paste(pil, (s, 0))
        s += pil.width
    return new


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    Ref: https://github.com/Sense-X/Co-DETR/
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_voc_dict(voc_dict):
    """Parse voc dict."""
    class2index = {'hat': 0, 'person': 1}
    box_lst, id_lst = [], []
    for obj in voc_dict['annotation']['object']:
        id_lst.append(class2index[obj['name'].lower()])

        bndbox = obj['bndbox']
        box = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
        box_lst.append(list(map(int, box)))
    return box_lst, id_lst


def parse_voc_xml(node):
    """Parse xml data."""
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def parse_xml(xml_path):
    """Get label data with file path."""
    node = ET.parse(xml_path).getroot()
    return parse_voc_dict(parse_voc_xml(node))