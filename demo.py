# -*- coding: utf-8 -*-
"""Demo Code."""

import time
import torch
from PIL import Image
from models import get_model
from config import get_arguments
from utils.misc import get_logger, draw_box
from datasets.transforms import Transforms

def load_input(image_path):
    pil = Image.open(image_path).convert('RGB')
    img_transforms = Transforms('test')
    img, _ = img_transforms.transforms(pil, {})
    mask = torch.ones_like(img, dtype=torch.float32)[0]
    return img, mask, pil


if __name__ == '__main__':
    args = get_arguments()
    logger = get_logger(args.save_root / 'logs', args.save_root.stem)
    device = torch.device(args.device)
    img, mask, pil = load_input(args.data_root)

    model, _, postprocessor = get_model(args, device)
    ckpt = torch.load(args.eval_model_path,
                      map_location=device,
                      weights_only=False)
    logger.info("Load model weights")
    logger.info(model.load_state_dict(ckpt['state_dict']))
    model.eval()

    img_w, img_h = pil.size
    raw_size = torch.tensor([[img_h, img_w]], device=device)
    with torch.inference_mode():
        output = model(img.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
        result = postprocessor(output, raw_size)[0]
    
    for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
        if score > 0.5:
            pil = draw_box(pil, box, color='red' if label ==1 else 'blue')
    pil.save(args.save_root / 'result.png')

    logger.info(f"Save the test result as '{args.save_root / 'result.png'}'")