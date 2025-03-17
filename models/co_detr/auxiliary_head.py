# -*- coding: utf-8 -*-
"""ATSS Code."""
import torch
from torchvision.ops import box_iou
from utils.box_ops import box_cxcywh_to_xyxy

def get_aux_heads(args):
    heads = {}
    for key in args.aux_heads:    
        if key == 'atss':
            heads[key] = ATSS(args.atss_k)
            
        if key == 'faster_rcnn':
            heads[key] = FasterRCNN()
    return heads

class ATSS(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        # ATSS with a single scale outputs
        indices = [] # matching indices
        pred_boxes = outputs['pred_boxes']
        
        # image-wise for loop
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
            
            temp_indices = [[], []]
            for i, (gt_box, k_boxes)in enumerate(zip(gt_boxes, k_boxes_per_gt)):
                ious = box_iou(box_cxcywh_to_xyxy(gt_box[None, :]),
                               box_cxcywh_to_xyxy(k_boxes))
                
                # calculate iou threshold
                threshold = torch.mean(ious) + torch.std(ious)

                # for loop by positive candidnates(Cg)
                x1, y1, x2, y2 = box_cxcywh_to_xyxy(gt_box)
                for k, iou in enumerate(ious[0]):
                    cx, cy = k_boxes[k][:2]

                    # iou and spatial condition
                    if iou >= threshold and (x1 <= cx <= x2) and (y1 <= cy <= y2):
                        temp_indices[0].append(k_indices[i][k].item())
                        temp_indices[1].append(i)
            indices.append(temp_indices)

        return [(torch.tensor(i, dtype=torch.int64), torch.tensor(j, dtype=torch.int64)) for i, j in indices]
    

class FasterRCNN(torch.nn.Module):
    def __init__(self, pos_threshold=0.7, neg_threshold=0.3):
        super().__init__()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        pos_indices = []
        neg_indices = []
        pred_boxes = outputs['pred_boxes']
        for target, pred_box in zip(targets, pred_boxes):
            temp_indices = [[], []]
            gt_boxes = target['boxes'].to(pred_box.device)
            if gt_boxes.shape[0] == 0:
                pos_indices.append(temp_indices)
                neg_indices.append(torch.arange(pred_box.shape[0], dtype=torch.long))
                continue
            
            ious = box_iou(box_cxcywh_to_xyxy(gt_boxes),
                        box_cxcywh_to_xyxy(pred_box))
            neg_indices.append(torch.where(torch.max(ious, dim=0)[0] < self.neg_threshold)[0].cpu())
            
            # pos indices
            for i, iou in enumerate(ious):
                above_threshold = torch.where(iou > self.pos_threshold)[0].tolist()
                pos_index = set(above_threshold)
                pos_index.add(iou.argmax().item())
                
                temp_indices[0] += list(pos_index)
                temp_indices[1] += [i] * len(pos_index)
            pos_indices.append(temp_indices)
        pos_indices = [(torch.tensor(i, dtype=torch.int64), torch.tensor(j, dtype=torch.int64)) for i, j in pos_indices]
        return pos_indices, neg_indices
        