"""DN functinos."""

import torch
from utils.misc import inverse_sigmoid


def preprocessing_for_dn(targets,
                         query,
                         batch_size,
                         num_group,
                         d_model,
                         label_enc,
                         num_class,
                         label_noise_scale,
                         box_noise_scale,
                         training):
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

    num_obj_lst = [len(t['labels']) for t in targets]
    max_obj = max(num_obj_lst)

    dn_indices, dn_targets = [], []
    indicator1 = torch.ones([1, max_obj * num_group, 1], device=device).repeat(bs, 1, 1)
    noised_labels = torch.randint(0, num_class, size=(bs, max_obj * num_group), device=device)
    noised_boxes = torch.rand(bs, max_obj * num_group, 4, device=device)
    for i, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels']
        num_obj = len(labels)
        if num_obj > 0:
            if num_obj < max_obj:
                add_num = max_obj - num_obj
                idx = torch.randint(0, num_obj, (add_num,))
                add_boxes = boxes[idx]
                add_labels = labels[idx]
                boxes = torch.cat([boxes, add_boxes], dim=0)
                labels = torch.cat([labels, add_labels], dim=0)

            boxes = boxes.repeat(num_group, 1)
            labels = labels.repeat(num_group)

            # noise on the label
            _labels = labels.clone()
            p = torch.rand_like(_labels.float())
            noise_idx = torch.nonzero(p < label_noise_scale).view(-1)
            new_label = torch.randint_like(noise_idx, 0, num_class, device=device)
            noised_labels[i, noise_idx] = new_label

            # noise on the box
            _boxes = boxes.clone()
            diff = torch.zeros_like(_boxes)
            diff[:, :2] = _boxes[:, 2:] / 2
            diff[:, 2:] = _boxes[:, 2:]
            _boxes += torch.mul((torch.rand_like(_boxes) * 2 - 1.0), diff) * box_noise_scale
            _boxes.clamp_(min=0.0, max=1.0)
            noised_boxes[i] = _boxes

        idx = torch.arange(len(labels))
        dn_indices.append((idx, idx))
        dn_targets.append({'boxes': boxes.to(device), 'labels': labels.to(device)})

    noised_label_query = torch.cat([label_enc(noised_labels), indicator1], dim=2)
    dn_label_query = torch.cat([noised_label_query,
                                model_label_query.unsqueeze(0).repeat(bs, 1, 1)], dim=1)
    dn_box_query = torch.cat([inverse_sigmoid(noised_boxes),
                              query.unsqueeze(0).repeat(bs, 1, 1)], dim=1)

    # decoder's self-attention mask
    num_total_query = (max_obj * num_group) + num_query
    attention_mask = torch.zeros((1, 1, num_total_query, num_total_query), device=device)
    for i in range(num_group):
        attention_mask[:, :, i * max_obj:(i + 1) * max_obj, i * max_obj:(i + 1) * max_obj] = 1
        if i == num_group - 1:
            attention_mask[:, :, :, (i + 1) * max_obj:] = 1

    return (dn_box_query, dn_label_query), (dn_targets, dn_indices), attention_mask


def postprocessing_for_dn(outputs, targets, num_group):
    max_obj = max([len(t['labels']) for t in targets])
    matching_part = []
    denoising_part = []
    for i, output in enumerate(outputs):
        pred_logits = output['pred_logits']
        pred_boxes = output['pred_boxes']
        pivot = max_obj * num_group
        matching_part.append({
            'pred_logits': pred_logits[:, pivot:],
            'pred_boxes': pred_boxes[:, pivot:]
        })
        denoising_part.append({
            'pred_logits': pred_logits[:, :pivot],
            'pred_boxes': pred_boxes[:, :pivot],
        })

    return matching_part, denoising_part
