# -*- coding: utf-8 -*-
"""Main code."""

import wandb
from train import train_model
from evaluate import evaluate_model
from config import get_arguments
from utils.misc import get_logger

if __name__ == '__main__':
    args = get_arguments()
    logger = get_logger(args.save_root, args.save_root.stem)
    logger.debug('Check Arguments')
    for k, v in vars(args).items():
        logger.debug(f'{k} = {v}')

    # set wandb
    if args.wb_flag:
        run = wandb.init(project=args.wb_project,
                         job_type=args.mode,
                         name=args.wb_name,
                         notes=args.wb_notes,
                         tags=args.wb_tags,
                         config=args)

    # run apps
    if args.mode == 'train':
        train_model(args, logger)
    else:
        evaluate_model(args, logger)
