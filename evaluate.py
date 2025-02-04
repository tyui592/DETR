# -*- coding: utf-8 -*-
"""Evaluation Code."""

import torch
from tqdm import tqdm
from models import get_model
from datasets import get_dataloader


def evaluate_model(args, logger):
    """Evaluate a model."""
    device = torch.device(args.device)

    # set data configurations for evaluation.
    dataloader = get_dataloader(args, shuffle=False, drop_last=False)

    # load a trained model weights
    model, criterion, postprocessor = get_model(args, device)

    ckpt = torch.load(args.eval_model_path,
                      map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    logger.info((f"Load the model weights from '{args.eval_model_path}', "
                 f"Trained steps/epochs: {ckpt['step']}/{ckpt['epoch']}"))

    results = evaluate_step(model, dataloader, postprocessor, logger, args, device)

    # save evaluation results
    for file_id, s in results.items():
        file_path = args.save_root / f"{file_id}.txt"
        file_path.write_text(s)

    return None


@torch.no_grad()
def evaluate_step(model, dataloader, postprocessor, logger, args, device):
    """Evaluate with a dataloader."""
    def _to_str(boxes, probs, labels):
        s = []
        for label, prob, box in zip(labels, probs, boxes):
            x1, y1, x2, y2 = box
            s.append(f"{label} {prob} {x1} {y1} {x2} {y2}")
        return '\n'.join(s)

    res = {}
    for inputs, targets in tqdm(dataloader):
        with torch.inference_mode():
            outputs, _ = model(inputs['images'].to(device),
                               inputs['masks'].to(device))

        # use last layer's output
        results = postprocessor(outputs[-1], inputs['raw_sizes'].to(device))

        for result, target, file_id in zip(results, targets, inputs['file_ids']):
            res[file_id] = _to_str(result['boxes'], result['scores'], result['labels'])

    return res
