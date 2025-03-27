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

    time_meter = defaultdict(AverageMeter)
    criterion.set_summary()
    tictoc = time.time()
    for index, (inputs, targets) in enumerate(dataloader, 1):
        time_meter['data'].update(time.time() - tictoc)
        global_step += 1

        tictoc = time.time()
        outputs = model(inputs['images'].to(device), inputs['masks'].to(device), targets)
        time_meter['forward'].update(time.time() - tictoc)

        # loss calculation
        tictoc = time.time()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total_loss = criterion(outputs=outputs, targets=targets)
        time_meter['loss_calc'].update(time.time() - tictoc)
        
        # backwarding
        tictoc = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        time_meter['backward'].update(time.time() - tictoc)

        if index % print_interval == 0:
            logging_losses(index=index,
                           step=global_step,
                           n_step=len(dataloader),
                           epoch=epoch,
                           summary=criterion.summary,
                           logger=logger,
                           args=args)
            logging_losses_debug(index=index,
                                 n_step=len(dataloader),
                                 epoch=epoch,
                                 summary=criterion.summary_debug,
                                 logger=logger,
                                 args=args)
        tictoc = time.time()

    # save predictions
    results = postprocessor(outputs['model'][-1], inputs['sizes'].to(device))
    pil = draw_outputs(inputs, targets, results)
    pil.save(args.save_root / 'logs'/ f'{epoch:03d}.png')
    logging_speeds(step=global_step,
                   epoch=epoch,
                   summary=time_meter,
                   logger=logger,
                   args=args)
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


def logging_losses(index, step, n_step, epoch, summary, logger, args):
    if args.wb_flag:
        wb_loss_log = {}

    loss_logs = []
    for key, value in summary.items():
        loss_logs.append(f"{key}: {value.avg:1.4f}")
        if args.wb_flag:
            wb_loss_log[key] = value.avg
    loss_summary = ', '.join(loss_logs)

    logger.info((
        f"Epoch: {epoch}/{args.epochs}, "
        f"Iter: {index}/{n_step}, "
        f"Loss: [{loss_summary}]"
        ))
    if args.wb_flag:
        wandb.log(wb_loss_log, step=step)
    return None

def logging_losses_debug(index, n_step, epoch, summary, logger, args):
    loss_logs = []
    for key, value in summary.items():
        loss_logs.append(f"{key}: {value.avg:1.4f}")
    loss_summary = ', '.join(loss_logs)

    logger.debug((
        f"Epoch: {epoch}/{args.epochs}, "
        f"Iter: {index}/{n_step}, "
        f"Loss: [{loss_summary}]"
        ))
    return None


def logging_speeds(step, epoch, summary, logger, args):
    speed_logs = []
    for key, value in summary.items():
        speed_logs.append(f"{key}: {value.avg*1000:6.2f}")
    speed_summary = ', '.join(speed_logs)

    logger.info((f"Step: {step}, "
                 f"Epoch: {epoch}/{args.epochs}, "
                 f"Speed(ms): [{speed_summary}]"))
    return None