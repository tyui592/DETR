# -*- coding: utf-8 -*-
"""Loss Code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from models.matcher import HungarianMatcher
from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy


def get_dab_detr_criterion(args, device):
    """Get a loss criterion with the arguments."""
    matcher = HungarianMatcher(cls_loss=args.cls_loss,
                               cost_class=args.cls_match_weight,
                               cost_bbox=args.l1_match_weight,
                               cost_giou=args.giou_match_weight,
                               focal_gamma=args.focal_gamma,
                               focal_alpha=args.focal_alpha)

    criterion = Criterion(matcher=matcher,
                          n_cls=args.n_cls,
                          cls_loss=args.cls_loss,
                          aux_flag=args.return_intermediate,
                          noobj_weight=args.noobj_cls_weight,
                          focal_gamma=args.focal_gamma,
                          focal_alpha=args.focal_alpha).to(device)

    return criterion


class Criterion(nn.Module):
    """Criterion Class."""

    def __init__(self,
                 matcher,
                 n_cls: int = 2,
                 cls_loss: str = 'ce',
                 aux_flag: bool = True,
                 noobj_weight: float = 0.1,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25):
        """Initialize."""
        super().__init__()
        self.matcher = matcher
        self.n_cls = n_cls
        self.aux_flag = aux_flag

        if cls_loss == 'ce':
            # class weights
            weight = torch.ones(n_cls + 1)
            weight[-1] = noobj_weight
            self.cls_criterion = CrossEntropy(weight=weight)
        else:
            self.cls_criterion = FocalLoss(alpha=focal_alpha,
                                           gamma=focal_gamma,
                                           n_cls=n_cls)

    def forward(self, outputs, targets):
        """Forward function."""
        num_boxes = sum(len(t['labels']) for t in targets)

        if not self.aux_flag:
            outputs = outputs[-1:]

        losses = defaultdict(list)
        # calculate losses per each decoder layer
        for output in outputs:
            indices = self.matcher(output, targets)

            # box loss
            l1_loss, giou_loss = self.calc_box_loss(output, targets, indices, num_boxes)
            losses['l1_loss'].append(l1_loss)
            losses['giou_loss'].append(giou_loss)

            # class loss
            cls_loss = self.calc_cls_loss(output, targets, indices, num_boxes)
            losses['cls_loss'].append(cls_loss)

        return losses

    def calc_box_loss(self, outputs, targets, indices, num_boxes):
        """Calculate box loss."""
        idx = self._get_src_permutation_idx(indices)

        selected_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)])

        l1_loss = F.l1_loss(selected_boxes, target_boxes, reduction='none')
        l1_loss = l1_loss.sum() / num_boxes

        giou_loss = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(selected_boxes),
            box_cxcywh_to_xyxy(target_boxes))
        )
        giou_loss = giou_loss.sum() / num_boxes

        return l1_loss, giou_loss

    def calc_cls_loss(self, outputs, targets, indices, num_boxes):
        """Calculate class loss."""
        idx = self._get_src_permutation_idx(indices)

        target_cls = torch.full(outputs['pred_logits'].shape[:2],
                                self.n_cls,
                                device=outputs['pred_logits'].device)
        target_cls_o = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)])
        target_cls[idx] = target_cls_o

        loss = self.cls_criterion(outputs['pred_logits'].transpose(1, 2), target_cls, num_boxes)
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class CrossEntropy(nn.Module):
    """CrossEntropy Class."""

    def __init__(self, weight):
        """Initialize."""
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets, num_boxes):
        """Forward function."""
        return self.criterion(inputs, targets)


class FocalLoss(nn.Module):
    """FocalLoss Class."""

    def __init__(self, alpha, gamma, n_cls):
        """Initialize."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.n_cls = n_cls

    def forward(self, inputs, targets, num_boxes):
        """Forward function."""
        inputs = inputs.transpose(1, 2)
        onehot = F.one_hot(targets, self.n_cls + 1)[:, :, :-1].to(inputs.device).float()
        loss = sigmoid_focal_loss(inputs, onehot, self.alpha, self.gamma)
        return loss / num_boxes * inputs.shape[1]


# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/segmentation.py#L196
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()
