"""DN functinos."""

import torch
import torchvision
from collections import defaultdict
from utils.misc import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy


def generate_noised_neg_boxes(existing_boxes,
                              num_boxes,
                              iou_threshold=0.2,
                              xy_noise_scale=1.5,
                              wh_noise_scale=0.2,
                              max_attempts=100):
    result = []
    attempt = 0
    boxes = existing_boxes.clone()
    boxes = boxes.repeat(100, 1)
    diff = torch.zeros_like(boxes)
    diff[:, :2] = (boxes[:, 2:] / 2) * xy_noise_scale
    diff[:, 2:] = boxes[:, 2:] * wh_noise_scale
    while len(result) < num_boxes and attempt < max_attempts:
        # Make neg boxes by add noise
        boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff)
        boxes.clamp_(min=0.0, max=1.0)

        # To avoid every GTs.
        iou_with_eboxes = torchvision.ops.box_iou(
            box_cxcywh_to_xyxy(boxes),
            box_cxcywh_to_xyxy(existing_boxes)
        )
        ious, _ = iou_with_eboxes.max(dim=1)
        idx = ious < iou_threshold
        selected_boxes = boxes[idx]
        result += selected_boxes.tolist()
        attempt += 1
    return torch.tensor(result[:num_boxes])

def make_cdn_query(targets: list = None,
                   bs: int = 4,
                   num_group: int = 5,
                   label_enc: torch.nn.Embedding = None,
                   num_class: int = 2,
                   label_noise_scale: float = 0.2,
                   box_noise_scale: float = 0.4,
                   num_cdn_query: int = 100,
                   add_neg_query: bool = True,
                   device = torch.device('cpu')):
    """Make noised positive and negative queies with targets.
    
    """
    # label query(embedding) of model(learnable) query
    if num_cdn_query > 0:
        max_obj = num_cdn_query # fix number of noised query
    else:
        # adpatively make noised queries by maximum number of object in batch
        num_obj_lst = [len(t['labels']) for t in targets]
        max_obj = max(num_obj_lst)

    cdn_indices, cdn_targets = [], []
    noised_labels = torch.randint(0, num_class, size=(bs, num_group, max_obj), device=device)
    noised_boxes = torch.rand(bs, num_group, max_obj, 4, device=device)

    if add_neg_query:
        # neg_labels = torch.randint(0, num_class, size=(bs, num_group, max_obj), device=device)
        neg_labels = torch.full((bs, num_group, max_obj), num_class, device=device)
        neg_boxes = torch.rand(bs, num_group, max_obj, 4, device=device) # dummy
    
    for i, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels']
        num_obj = len(labels)
        
        # When the image has a foreground object
        if num_obj > 0:
            # Create negative boxes that does not overlap the GT Box.
            if add_neg_query:
                _neg_boxes = generate_noised_neg_boxes(boxes,
                                                       num_boxes=max_obj * num_group).to(device)
                neg_boxes[i] = _neg_boxes.view(num_group, max_obj, -1)

            # If the number of objects is less than the target(num_cdn_query) number, more objects are sampled.
            if num_obj < max_obj:
                add_num = max_obj - num_obj
                idx = torch.randint(0, num_obj, (add_num,))
                add_boxes = boxes[idx]
                add_labels = labels[idx]
                boxes = torch.cat([boxes, add_boxes], dim=0)
                labels = torch.cat([labels, add_labels], dim=0)
            elif num_obj > max_obj:
                idx = torch.randint(0, num_obj, (max_obj,))
                boxes = boxes[idx]
                labels = labels[idx]
            
            boxes = boxes.repeat(num_group, 1).view(num_group, max_obj, -1)
            labels = labels.repeat(num_group).view(num_group, max_obj)

            # noise on the label
            _labels = labels.clone()
            p = torch.rand_like(_labels.float())
            noise_idx = torch.nonzero(p.view(-1) < label_noise_scale).view(-1)
            new_label = torch.randint_like(noise_idx, 0, num_class, device=device)
            noised_labels[i].view(-1)[noise_idx] = new_label

            # noise on the box
            _boxes = boxes.clone()
            diff = torch.zeros_like(_boxes)
            diff[:, :, :2] = _boxes[:, :, 2:] / 2
            diff[:, :, 2:] = _boxes[:, :, 2:]
            _boxes += torch.mul((torch.rand_like(_boxes) * 2 - 1.0), diff) * box_noise_scale
            _boxes.clamp_(min=0.0, max=1.0)
            noised_boxes[i] = _boxes

            # flatten gt boxes
            boxes = boxes.flatten(0, 1)
            labels = labels.flatten(0, 1)

        # indices for loss calculation
        num_pos = max_obj if num_obj != 0 else num_obj
        idx = torch.arange(num_pos).unsqueeze(0).repeat(num_group, 1)
        offset0 = torch.arange(num_group).unsqueeze(1) * num_pos * (2 if add_neg_query else 1)
        offset1 = torch.arange(num_group).unsqueeze(1) * num_pos
        indices0 = (idx + offset0).view(-1)
        indices1 = (idx + offset1).view(-1)
        cdn_indices.append((indices0, indices1))

        # targets for loss calculation
        cdn_targets.append({'boxes': boxes.to(device),
                           'labels': labels.to(device)})

    if add_neg_query:
        cdn_labels = torch.cat([noised_labels, neg_labels], dim=2)
        cdn_boxes = torch.cat([noised_boxes, neg_boxes], dim=2)
    else:
        cdn_labels = noised_labels
        cdn_boxes = noised_boxes
        
    cdn_labels = cdn_labels.flatten(1, 2)
    cdn_boxes = cdn_boxes.flatten(1, 2)
    
    indicator1 = torch.ones([1, 1, 1], device=device).repeat(bs, cdn_labels.shape[1], 1)
    cdn_label_query = torch.cat([label_enc(cdn_labels), indicator1], dim=-1)

    cdn_box_query = inverse_sigmoid(cdn_boxes)
    
    # decoder's self-attention mask
    num_total_query = cdn_label_query.shape[1]
    group_size = max_obj
    if add_neg_query:
        group_size *= 2
    attn_mask = torch.zeros((num_total_query, num_total_query), device=device)
    for i in range(num_group):
        attn_mask[i * group_size:(i + 1) * group_size, i * group_size:(i + 1) * group_size] = 1
    
    cdn = {
        'anchor_query': cdn_box_query,
        'label_query': cdn_label_query,
        'targets': cdn_targets,
        'indices': cdn_indices,
        'attn_mask': attn_mask,
        }
    return cdn

def split_outputs(outputs, sizes, labels):
    parts = defaultdict(list)
    for output in outputs:
        pred_logits_lst = output['pred_logits'].split(sizes, dim=1)
        pred_boxes_lst = output['pred_boxes'].split(sizes, dim=1)
        for label, pred_logits, pred_boxes in zip(labels, pred_logits_lst, pred_boxes_lst):
            parts[label].append({
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            })
    return parts