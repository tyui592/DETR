"""DN functinos."""

import torch
import torchvision
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
        boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff)
        boxes.clamp_(min=0.0, max=1.0)

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


def preprocessing_for_dino(targets: list = None,
                           query: torch.tensor = None,
                           batch_size: int = 4,
                           num_group: int = 5,
                           d_model: int = 256,
                           label_enc: torch.nn.Embedding = None,
                           num_class: int = 2,
                           label_noise_scale: float = 0.2,
                           box_noise_scale: float = 0.4,
                           num_dn_query: int = 100,
                           add_neg_query: bool = True,
                           training: bool = True):
    bs = batch_size
    num_query = query.shape[0]
    device = query.device

    # model label query
    model_label_query = label_enc(torch.tensor(num_class, device=device)).repeat(num_query, 1)
    indicator0 = torch.zeros([num_query, 1], device=device)
    model_label_query = torch.cat([model_label_query, indicator0], dim=1)

    if not training:
        model_label_query = model_label_query.unsqueeze(0).repeat(bs, 1, 1)
        model_box_query = query.unsqueeze(0).repeat(bs, 1, 1)
        return (model_box_query, model_label_query), None, None

    if num_dn_query > 0:
        max_obj = num_dn_query # fix number of dn query for efficiency
    else:
        # adpatively make dn queries
        num_obj_lst = [len(t['labels']) for t in targets]
        max_obj = max(num_obj_lst)

    dn_indices, dn_targets = [], []
    indicator1 = torch.ones([1, 1, 1], device=device).repeat(bs, 1, 1)
    noised_labels = torch.randint(0, num_class, size=(bs, num_group, max_obj), device=device)
    noised_boxes = torch.rand(bs, num_group, max_obj, 4, device=device)

    if add_neg_query:
        neg_labels = torch.randint(0, num_class, size=(bs, num_group, max_obj), device=device)
        neg_boxes = torch.rand(bs, num_group, max_obj, 4, device=device)

    for i, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels']
        num_obj = len(labels)
        if num_obj > 0:
            if add_neg_query:
                _neg_boxes = generate_noised_neg_boxes(boxes,
                                                       num_boxes=max_obj * num_group).to(device)
                neg_boxes[i] = _neg_boxes.view(num_group, max_obj, -1)

            # If the number of objects is less than the target number, additional objects are sampled.
            if num_obj < max_obj:
                add_num = max_obj - num_obj
                idx = torch.randint(0, num_obj, (add_num,))
                add_boxes = boxes[idx]
                add_labels = labels[idx]
                boxes = torch.cat([boxes, add_boxes], dim=0)
                labels = torch.cat([labels, add_labels], dim=0)
            # If the number of objects is greater than the target number, some of them are sampled.
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
        offset0 = torch.arange(num_group).unsqueeze(1) * num_pos * 2
        offset1 = torch.arange(num_group).unsqueeze(1) * num_pos
        indices0 = (idx + offset0).view(-1)
        indices1 = (idx + offset1).view(-1)
        dn_indices.append((indices0, indices1))

        # targets for loss calculation
        dn_targets.append({'boxes': boxes.to(device),
                           'labels': labels.to(device)})

    if add_neg_query:
        cdn_labels = torch.cat([noised_labels, neg_labels], dim=2)
        cdn_boxes = torch.cat([noised_boxes, neg_boxes], dim=2)
    else:
        cdn_labels = noised_labels
        cdn_boxes = noised_boxes
    cdn_labels = cdn_labels.flatten(1, 2)
    cdn_boxes = cdn_boxes.flatten(1, 2)

    # concatenate group part and matching part
    group_label_query = torch.cat([label_enc(cdn_labels),
                                   indicator1.repeat(1, cdn_labels.shape[1], 1)], dim=2)
    cdn_label_query = torch.cat([group_label_query,
                                 model_label_query.unsqueeze(0).repeat(bs, 1, 1)], dim=1)

    group_box_query = inverse_sigmoid(cdn_boxes)
    cdn_box_query = torch.cat([group_box_query,
                               query.unsqueeze(0).repeat(bs, 1, 1)], dim=1)

    # decoder's self-attention mask
    num_total_query = cdn_label_query.shape[1]
    group_size = max_obj
    if add_neg_query:
        group_size *= 2
    attention_mask = torch.zeros((1, 1, num_total_query, num_total_query), device=device)
    for i in range(num_group):
        attention_mask[:, :, i * group_size:(i + 1) * group_size, i * group_size:(i + 1) * group_size] = 1
        if i == num_group - 1:
            attention_mask[:, :, :, (i + 1) * group_size:] = 1

    return (cdn_box_query, cdn_label_query), (dn_targets, dn_indices), attention_mask


def postprocessing_for_dino(outputs, targets, num_query):
    matching_part = []
    denoising_part = []
    for i, output in enumerate(outputs):
        pred_logits = output['pred_logits']
        pred_boxes = output['pred_boxes']
        matching_part.append({
            'pred_logits': pred_logits[:, -num_query:],
            'pred_boxes': pred_boxes[:, -num_query:]
        })
        denoising_part.append({
            'pred_logits': pred_logits[:, :-num_query],
            'pred_boxes': pred_boxes[:, :-num_query],
        })

    return matching_part, denoising_part

@torch.no_grad()
def get_query(pred_boxes, pred_logits, mask, topk):
    """
    pred_boxes, pred_logits: raw outout of the encoder
    mask: image padding mask
    topk: for encoder query
    """
    N = pred_logits.shape[0]
    pred_logits.masked_fill_(mask.squeeze().unsqueeze(-1) == 0, float('-inf'))
    max_logits, max_labels = torch.max(pred_logits, dim=2)
    _, topk_indices = torch.topk(max_logits, k=topk, dim=1)
    batch_indices = torch.arange(end=N).unsqueeze(-1).expand(N, topk)
    
    boxes = pred_boxes[(batch_indices, topk_indices)]
    labels = max_labels[(batch_indices, topk_indices)]
    return boxes, labels

def postprocessing_for_enc_query(boxes, labels, label_enc, input_box_query, input_label_query, attn_mask):
    """Make decoder input query and attn. mask
    
    boxes: boxes from encoder
    labels: labels from encoder
    label_enc: label embedding (nn.Embedding), 0 ~ Num Class + 1(for learnable query)
    input_box_query: current input query(denoised query + negative query + learnable query)
    input_label_query: current input query(denoised query + negative query + learnable query)
    attn_mask: attention mask for the cross-attnetion
    """
    N, C = labels.shape
    device = input_box_query.device
    
    # make label query
    enc_label_query = label_enc(labels.to(device))
    indicator1 = torch.ones([1, 1, 1], device=device).repeat(N, C, 1)
    enc_label_query = torch.cat([enc_label_query, indicator1], dim=-1)
    label_query = torch.cat([input_label_query, enc_label_query], dim=1)
    
    # make box query
    enc_box_query = inverse_sigmoid(boxes).to(device)
    box_query = torch.cat([input_box_query, enc_box_query], dim=1)
    
    # edit attn_mask
    Ci = input_box_query.shape[1]
    num_dn_query = int(sum(input_label_query[0, :, -1]).item())    
    new_attn_mask = torch.zeros((Ci + C, Ci + C), dtype=attn_mask.dtype, device=device)
    new_attn_mask[:Ci, :Ci] = attn_mask
    new_attn_mask[Ci:, num_dn_query:] = 1
    
    return label_query, box_query, new_attn_mask