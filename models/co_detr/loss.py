# -*- coding: utf-8 -*-
"""Loss Code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from models.matcher import HungarianMatcher
from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from utils.misc import AverageMeter
from models.atss import ATSS, ModifiedATSS


def get_co_detr_criterion(args, device):
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
                          l1_weight=args.l1_loss_weight,
                          giou_weight=args.giou_loss_weight,
                          cls_weight=args.cls_loss_weight,
                          focal_gamma=args.focal_gamma,
                          focal_alpha=args.focal_alpha,
                          atss_mode=args.atss_mode,
                          atss_k=args.atss_k).to(device)

    return criterion


class Criterion(nn.Module):
    """Criterion Class."""

    def __init__(self,
                 matcher,
                 n_cls: int = 2,
                 cls_loss: str = 'ce',
                 aux_flag: bool = True,
                 noobj_weight: float = 0.1,
                 l1_weight: float = 1.0,
                 giou_weight: float = 1.0,
                 cls_weight: float = 1.0,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25,
                 atss_mode: str = None,
                 atss_k: int = 20):
        """Initialize.

        atss_mode: 'none', 'atss', 'matss'
        """
        super().__init__()
        self.matcher = matcher
        self.n_cls = n_cls
        self.aux_flag = aux_flag
        self.cls_loss = cls_loss
        self.loss_weights = {'l1': l1_weight,
                             'giou': giou_weight,
                             'cls': cls_weight}

        if atss_mode == 'atss':
            self.atss = ATSS(atss_k, 'mean+std')

        elif atss_mode == 'atss_mean':
            self.atss = ATSS(atss_k, 'mean')

        elif atss_mode == 'matss':
            self.atss = ModifiedATSS(atss_k, n=atss_k//2)

        if self.cls_loss == 'ce':
            # class weights
            weight = torch.ones(self.n_cls + 1)
            weight[-1] = noobj_weight
            self.cls_criterion = CrossEntropy(weight=weight)
        else:
            self.cls_criterion = FocalLoss(alpha=focal_alpha,
                                           gamma=focal_gamma,
                                           n_cls=self.n_cls)
            
        self.set_summary()
        return None
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def set_summary(self):
        self.summary = defaultdict(AverageMeter)        
            
    @torch.no_grad()
    def calc_cardinality(self, outputs, targets):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return card_err.item()
    
    @torch.no_grad()
    def calc_top1_accuracy(self, logits, labels):
        top1_labels = logits.argmax(-1)
        acc = ((top1_labels == labels).sum()) / len(labels)
        return acc.item()
        
    def calc_box_loss(self, outputs, targets, indices):
        """Calculate box loss."""
        idx = self._get_src_permutation_idx(indices)
        num_boxes = max(len(idx[0]), 1)

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

    def calc_cls_loss(self, outputs, targets, indices, check_acc=False):
        """Calculate class loss."""
        idx = self._get_src_permutation_idx(indices)
        num_boxes = max(len(idx[0]), 1)

        target_cls = torch.full(outputs['pred_logits'].shape[:2],
                                self.n_cls,
                                device=outputs['pred_logits'].device)
        target_cls_o = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)])
        target_cls[idx] = target_cls_o
        
        loss = self.cls_criterion(outputs['pred_logits'].transpose(1, 2), target_cls, num_boxes)
        
        if check_acc:
            top1_acc = self.calc_top1_accuracy(outputs['pred_logits'][idx], target_cls_o)
            self.summary['top1_acc'].update(top1_acc, n=len(target_cls_o))
            
        return loss
    
    def calc_total_loss(self, losses):
        """Calcualte the total loss"""
        
        # Calculate three losses (l1, giou, cls)
        total_losses = defaultdict(list)
        for key, values in losses.items():
            if 'l1' in key:
                total_losses['l1'] += values
            elif 'giou' in key:
                total_losses['giou'] += values
            elif 'cls' in key:
                total_losses['cls'] += values
        
        # Weighted sum
        total_loss = 0
        for key, weight in self.loss_weights.items():
            mean = sum(total_losses[key]) / len(total_losses[key])
            
            self.summary[key].update(mean, n=len(total_losses[key]))
            total_loss += mean * weight
            
        self.summary['total'].update(total_loss.item(),
                                     n=len(total_losses[key]))
        return total_loss
    
    def forward(self, outputs: dict, targets: list):
        """Forward function.

        output['model']: decoder's output with learnable(model) queries
        output['denosing']: decoder's output with noised queries
        output['encoder']: decoder's output with encoder queries(two-stage)
        output['enc_outputs']: encoder's output with encoder features
        """
        # loss calculation per parts
        losses = defaultdict(list)
        for key in ['model', 'cdn']:
            _outputs = outputs[key]
            if not self.aux_flag:
                _outputs = _outputs[-1:]
            for output in _outputs:
                if key == 'cdn':
                    indices = outputs['cdn_indices']
                    _targets = outputs['cdn_targets']
                else:
                    indices =  self.matcher(output, targets)
                    _targets = targets
                
                l1_loss, giou_loss = self.calc_box_loss(output, _targets, indices)
                cls_loss = self.calc_cls_loss(output, _targets, indices, key=='model')

                losses[f'{key}_l1_loss'].append(l1_loss)
                losses[f'{key}_giou_loss'].append(giou_loss)
                losses[f'{key}_cls_loss'].append(cls_loss)
                
                if key == 'model' and self.cls_loss == 'ce':
                    cardinality = self.calc_cardinality(output, _targets)
                    self.summary['cardinality'].update(cardinality)
                    
        # first-stage outputs
        for output in outputs['first_stage']:
            if hasattr(self, 'atss'):
                indices = self.atss(output, targets)
            else:
                indices = self.matcher(output, targets)

            l1_loss, giou_loss = self.calc_box_loss(output, targets, indices)
            cls_loss = self.calc_cls_loss(output, targets, indices)
            losses['first_l1_loss'].append(l1_loss)
            losses['first_giou_loss'].append(giou_loss)
            losses['first_cls_loss'].append(cls_loss)

        total_loss = self.calc_total_loss(losses)

        return total_loss


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
