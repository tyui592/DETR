# -*- coding: utf-8 -*-
"""ATSS Code."""
import torch
from torchvision.ops import box_iou
from utils.box_ops import box_cxcywh_to_xyxy

def get_atss(atss_mode, atss_k):
    atss = None
    if atss_mode == 'atss':
        atss = ATSS(atss_k, 'mean+std')

    elif atss_mode == 'atss_mean':
        atss = ATSS(atss_k, 'mean')

    elif atss_mode == 'matss':
        atss =  ModifiedATSS(atss_k, atss_k//2)
    return atss

class ATSS(torch.nn.Module):
    def __init__(self, k, threshold='mean+std'):
        super().__init__()
        self.k = k
        self.threshold = threshold
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = [] # matching indices
        pred_boxes = outputs['pred_boxes']
        for target, pred_box in zip(targets, pred_boxes):
            temp_indices = [[], []]
            gt_boxes = target['boxes'].to(pred_box.device)
            if gt_boxes.shape[0] == 0:
                indices.append(temp_indices)
                continue
            distances = torch.cdist(gt_boxes[: ,:2], pred_box[:, :2], p=2.0)
            
            # select closest K anchors
            _, k_indices = torch.topk(distances, k=self.k, dim=1, largest=False)

            k_boxes_per_gt = pred_box[k_indices]
            # for loop by GT box(g)
            temp_indices = [[], []]
            for i, (gt_box, k_boxes)in enumerate(zip(gt_boxes, k_boxes_per_gt)):
                x1, y1, x2, y2 = box_cxcywh_to_xyxy(gt_box)

                # compute ious
                ious = box_iou(box_cxcywh_to_xyxy(gt_box[None, :]),
                               box_cxcywh_to_xyxy(k_boxes))
                
                # calculate iou threshold
                if self.threshold == 'mean':
                    threshold = torch.mean(ious)
                elif self.threshold == 'mean+std':
                    threshold = torch.mean(ious) + torch.std(ious)

                # for loop by positive candidnates(Cg)
                for k, iou in enumerate(ious[0]):
                    cx, cy = k_boxes[k][:2]

                    # iou and spatial condition
                    if iou >= threshold and (x1 <= cx <= x2) and (y1 <= cy <= y2):
                        temp_indices[0].append(k_indices[i][k].item())
                        temp_indices[1].append(i)
            indices.append(temp_indices)

        return [(torch.tensor(i, dtype=torch.int64), torch.tensor(j, dtype=torch.int64)) for i, j in indices]

class ModifiedATSS(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.k = k
        self.n = n

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = [] # matching indices
        pred_boxes = outputs['pred_boxes']
        for target, pred_box in zip(targets, pred_boxes):
            temp_indices = [[], []]
            gt_boxes = target['boxes'].to(pred_box.device)

            if gt_boxes.shape[0] == 0:
                indices.append(temp_indices)
                continue
            distances = torch.cdist(gt_boxes, pred_box, p=2.0)
            # select closest K anchors
            _, k_indices = torch.topk(distances, k=self.k, dim=1, largest=False)

            k_boxes_per_gt = pred_box[k_indices]
            # for loop by GT box(g)
            temp_indices = [[], []]
            for i, (gt_box, k_boxes)in enumerate(zip(gt_boxes, k_boxes_per_gt)):
                # ious
                ious = box_iou(box_cxcywh_to_xyxy(gt_box[None, :]),
                               box_cxcywh_to_xyxy(k_boxes))

                _, ks = torch.topk(ious, self.n)
                for k in ks[0]:
                    temp_indices[0].append(k_indices[i][k].item())
                    temp_indices[1].append(i)            
            indices.append(temp_indices)

        return [(torch.tensor(i, dtype=torch.int64), torch.tensor(j, dtype=torch.int64)) for i, j in indices]