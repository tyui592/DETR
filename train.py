# -*- coding: utf-8 -*-
"""Train Code."""

import time
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict

from models import get_model
from datasets import get_dataloader
from utils.misc import AverageMeter, draw_outputs


def train_model(args, logger):
    """Train a model."""
    device = torch.device(args.device)

    dataloader = get_dataloader(args)
    model, criterion, postprocessor = get_model(args, device)
    logger.debug('Load a model')
    logger.debug(model)
    optimizer, scheduler = get_optimizer(model, args)

    model.train()

    global global_step
    global_step = 0

    logger.info("Start training...")
    for epoch in range(args.epochs):
        train_step(model, dataloader, criterion, postprocessor, optimizer, logger, device, epoch, args)
        if scheduler is not None:
            scheduler.step()

        save_root = args.save_root / 'weights'
        save_root.mkdir(exist_ok=True)
        save_path = save_root / "checkpoint.pth"
        if (epoch + 1) % 5 == 0:
            save_path = save_root / f"{global_step:08d}.pth"
        torch.save({'step': global_step,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict()},
                   save_path)
    logger.info("End training!")

    return model


def train_step(model, dataloader, criterion, postprocessor, optimizer, logger, device, epoch, args):
    """Train one epoch."""
    global global_step

    if args.print_interval < 1:
        print_interval = int(len(dataloader) * args.print_interval)

    loss_meter = defaultdict(AverageMeter)
    time_meter = defaultdict(AverageMeter)
    tictoc = time.time()
    for index, (inputs, targets) in enumerate(dataloader, 1):
        time_meter['data'].update(time.time() - tictoc)
        global_step += 1

        tictoc = time.time()
        if args.require_target:
            outputs, _ = model(inputs['images'].to(device), inputs['masks'].to(device), targets)
        else:
            outputs, _ = model(inputs['images'].to(device), inputs['masks'].to(device))
        time_meter['forward'].update(time.time() - tictoc)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss calculation
        tictoc = time.time()
        losses = criterion(outputs=outputs, targets=targets)
        time_meter['loss_calc'].update(time.time() - tictoc)

        l1_loss = sum(losses['l1_loss']) / len(losses['l1_loss'])
        giou_loss = sum(losses['giou_loss']) / len(losses['giou_loss'])
        cls_loss = sum(losses['cls_loss']) / len(losses['cls_loss'])

        # total loss
        total_loss = (l1_loss * args.l1_loss_weight) \
            + (giou_loss * args.giou_loss_weight) \
            + (cls_loss * args.cls_loss_weight)

        tictoc = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        time_meter['backward'].update(time.time() - tictoc)

        loss_meter['l1'].update(l1_loss.item(), n=len(losses['l1_loss']))
        loss_meter['giou'].update(giou_loss.item(), n=len(losses['giou_loss']))
        loss_meter['cls'].update(cls_loss.item(), n=len(losses['cls_loss']))
        loss_meter['total'].update(total_loss.item(), n=len(losses['cls_loss']))
        if index % print_interval == 0:
            logger.info((f"Step: {global_step}, "
                         f"Epoch: {epoch}/{args.epochs}, "
                         f"Iter: {index}/{len(dataloader)}, "
                         f"Loss: [total: {loss_meter['total'].avg:1.4f}, "
                         f"box: {loss_meter['l1'].avg:1.4f}, "
                         f"iou: {loss_meter['giou'].avg:1.4f}, "
                         f"cls: {loss_meter['cls'].avg:1.4f}]"))
            if args.wb_flag:
                wandb.log({'box loss': loss_meter['l1'].avg,
                           'iou loss': loss_meter['giou'].avg,
                           'total loss': loss_meter['total'].avg,
                           'cls loss': loss_meter['cls'].avg},
                          step=global_step)
        tictoc = time.time()

    # save predictions
    if args.require_target:
        outputs = outputs[0]
    results = postprocessor(outputs[-1], inputs['sizes'].to(device))
    pil = draw_outputs(inputs, targets, results, args.topk)
    pil.save(args.save_root / 'training_image.png')

    logger.info((f"Step: {global_step}, "
                 f"Epoch: {epoch}/{args.epochs}, "
                 f"Loss: [total: {loss_meter['total'].avg:1.4f}, "
                 f"box: {loss_meter['l1'].avg:1.4f}, "
                 f"iou: {loss_meter['giou'].avg:1.4f}, "
                 f"cls: {loss_meter['cls'].avg:1.4f}]"))
    logger.info((f"Step: {global_step}, "
                 f"Epoch: {epoch}/{args.epochs}, "
                 f"Time(ms): [data: {time_meter['data'].avg*1000:6.2f}, "
                 f"forward: {time_meter['forward'].avg*1000:6.2f}, "
                 f"loss: {time_meter['loss_calc'].avg*1000:6.2f}, "
                 f"backward: {time_meter['backward'].avg*1000:6.2f}]\n"))

    return None


def get_optimizer(model, args):
    """Get optimizer."""
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_milestone is not None:
        scheduler = MultiStepLR(optimizer=optimizer,
                                milestones=args.lr_milestone,
                                gamma=args.lr_gamma)
    return optimizer, scheduler
