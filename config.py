# -*- coding: utf-8 -*-
"""Configurations."""

import argparse
from pathlib import Path
from utils.misc import save_dict, load_dict, set_random_seed


def build_parser():
    """Get arguments from cmd."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',
                        type=int,
                        help='random seed',
                        default=777)

    parser.add_argument('--mode',
                        type=str,
                        choices=['train', 'test'],
                        default='train')

    parser.add_argument('--device',
                        type=str,
                        help='cpu, cuda:0 ~ N',
                        default='cuda:0')

    # Model
    parser.add_argument('--model',
                        type=str,
                        choices=['detr', 'conditional_detr', 'dab-detr',
                                 'dn-detr', 'dino-detr', 'co-detr'],
                        default='detr')

    parser.add_argument('--n_query',
                        type=int,
                        help='number of detector query',
                        default=300)

    parser.add_argument('--d_model',
                        type=int,
                        help='model dim = encoder_embed_dim',
                        default=256)

    parser.add_argument('--n_cls',
                        type=int,
                        help='number of class(0: no-helmet, 1: helmet)',
                        default=2)

    parser.add_argument('--n_heads',
                        type=int,
                        help='number of heads',
                        default=8)

    parser.add_argument('--d_ff',
                        type=int,
                        help='dim for feedforward',
                        default=2048)

    parser.add_argument('--p_drop',
                        type=float,
                        help='dropout prob',
                        default=0.0)

    parser.add_argument('--backbone',
                        type=str,
                        choices=['convnext_tiny',
                                 'resnet18', 'resnet34',
                                 'resnet50', 'resnet50_dc5'],
                        default='resnet50')

    parser.add_argument('--layer_index',
                        type=int,
                        help='for resnet backbone',
                        default=7)

    parser.add_argument('--n_encoder_layers',
                        type=int,
                        help='number of encoder layers',
                        default=6)

    parser.add_argument('--n_decoder_layers',
                        type=int,
                        help='number of decoder layers',
                        default=6)

    parser.add_argument('--activation',
                        type=str,
                        choices=['relu', 'gelu', 'linear'],
                        default='relu')

    parser.add_argument('--encoder_position_mode',
                        type=str,
                        help='positional encoding strategy',
                        choices=['add', 'cat', 'addcat'],
                        default='add')

    parser.add_argument('--decoder_sa_position_mode',
                        type=str,
                        help='positional encoding strategy',
                        choices=['add', 'cat', 'addcat'],
                        default='add')

    parser.add_argument('--decoder_ca_position_mode',
                        type=str,
                        help='positional encoding strategy',
                        choices=['add', 'cat', 'addcat'],
                        default='cat')

    parser.add_argument('--return_intermediate',
                        action='store_true',
                        help='for decoder learning with aux loss',
                        default=False)
    
    parser.add_argument('--pos_embedding',
                        type=str,
                        help='positional embedding method',
                        choices=['sine', 'sinev2', 'learned'],
                        default='sine')

    parser.add_argument('--temperature',
                        type=float,
                        default=10000,
                        help='temperature of positional encoding')

    # Conditional DETR
    parser.add_argument('--query_scale_mode',
                        type=str,
                        help='(Conditional DETR), transformer T mode',
                        choices=['diag', 'identity', 'scalar'],
                        default='diag')

    # DAB-DETR
    parser.add_argument('--num_pattern',
                        type=int,
                        help='(DAB-DETR), Number of patterns for anchors',
                        default=0)

    parser.add_argument('--modulate_wh_attn',
                        action='store_true',
                        help='(DAB-DETR) Modulate attn with Scale(HW)',
                        default=False)

    parser.add_argument('--iter_update',
                        action='store_true',
                        help='(DAB-DETR) Iteratively update anchors',
                        default=False)

    parser.add_argument('--transformer_activation',
                        type=str,
                        help='(DAB-DETR) Iteratively update anchors',
                        choices=['relu', 'gelu', 'linear', 'prelu'],
                        default='relu')

    parser.add_argument('--fix_init_xy',
                        action='store_true',
                        help='(DAB-DETR) Use randomly initialized xy points',
                        default=False)

    # DN-DETR, DINO-DETR
    parser.add_argument('--num_group',
                        type=int,
                        help='(DN-DETR), Number of groups',
                        default=3)

    parser.add_argument('--box_noise_scale',
                        type=float,
                        help='(DN-DETR), Box noise scale',
                        default=0.4)

    parser.add_argument('--label_noise_scale',
                        type=float,
                        help='(DN-DETR), Box noise scale',
                        default=0.2)
    # DINO-DETR
    parser.add_argument('--num_dn_query',
                        type=int,
                        help='(DINO-DETR), Number of denoised query',
                        default=100)

    parser.add_argument('--add_neg_query',
                        action='store_true',
                        help='(DINO) Add Negative query in the group',
                        default=False)
    # Two stage
    parser.add_argument('--two_stage_mode',
                        type=str,
                        choices=['none', 'static', 'pure', 'mix'],
                        help='"none": not use, "add": concat encoders output with learnable query and cdn query during training, "pure" and "mix": use encoders output as model query',
                        default='none')
        
    parser.add_argument('--two_stage_share_head',
                        action='store_true',
                        default=False)
    
    parser.add_argument('--num_encoder_query',
                        type=int,
                        default=100)


    # Data
    parser.add_argument('--dataset',
                        type=str,
                        choices=['shwd'],
                        default='shwd')

    parser.add_argument('--data_root',
                        type=Path,
                        help='data root',
                        default='../VOC2028/')

    parser.add_argument('--image_set',
                        type=str,
                        help='image set for experiment(train)',
                        choices=['train', 'trainval', 'test'],
                        default='trainval')

    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=0.1)

    parser.add_argument('--num_workers',
                        type=int,
                        help='for dataloader',
                        default=4)

    parser.add_argument('--pin_memory',
                        action='store_true',
                        help='for dataloader',
                        default=False)

    # Training
    parser.add_argument('--image_size',
                        type=int,
                        help='image size',
                        default=608)

    parser.add_argument('--batch_size',
                        type=int,
                        default=8)

    parser.add_argument('--epochs',
                        type=int,
                        default=60)

    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0001)

    parser.add_argument('--lr',
                        type=float,
                        default=0.0001)

    parser.add_argument('--lr_backbone',
                        type=float,
                        help='encoder lr, 0: no train',
                        default=0.00001)

    parser.add_argument('--lr_milestone',
                        type=int,
                        nargs='+',
                        default=[50])

    parser.add_argument('--lr_gamma',
                        type=float,
                        default=0.1)
    # Matcher
    parser.add_argument('--cls_match_weight',
                        type=float,
                        default=2.0)
    
    parser.add_argument('--l1_match_weight',
                        type=float,
                        default=5.0)

    parser.add_argument('--giou_match_weight',
                        type=float,
                        default=2.0)
    
    # ATSS
    parser.add_argument('--atss_mode',
                        type=str,
                        choices=['none', 'atss', 'atss_mean', 'matss'],
                        default='none')
    
    parser.add_argument('--atss_k',
                        type=int,
                        default=20)

    # Loss
    parser.add_argument('--cls_loss',
                        type=str,
                        default='ce',
                        choices=['ce', 'focal'],
                        help='classification loss')

    parser.add_argument('--focal_gamma',
                        type=float,
                        default=2.0)

    parser.add_argument('--focal_alpha',
                        type=float,
                        help='pos vs neg balance factor, -1 no weighting',
                        default=0.25)

    parser.add_argument('--cls_loss_weight',
                        type=float,
                        default=2.0)
    
    parser.add_argument('--l1_loss_weight',
                        type=float,
                        default=5.0)

    parser.add_argument('--giou_loss_weight',
                        type=float,
                        default=2.0)

    parser.add_argument('--noobj_cls_weight',
                        type=float,
                        help='class weight',
                        default=0.1)

    # misc
    parser.add_argument('--save_root',
                        type=Path,
                        default='./model-store/ex01/')

    parser.add_argument('--topk',
                        type=int,
                        help='for visualize training process',
                        default=10)

    parser.add_argument('--print_interval',
                        type=float,
                        default=0.1,
                        help='0.1: print every 10 percent of one epoch iterations')

    # wandb
    parser.add_argument('--wb_flag',
                        action='store_true',
                        default=False,
                        help="Use wandb")

    parser.add_argument('--wb_project',
                        type=str,
                        default='detr')

    parser.add_argument('--wb_name',
                        type=str,
                        default=None)

    parser.add_argument('--wb_notes',
                        type=str,
                        default=None)

    parser.add_argument('--wb_tags',
                        type=str,
                        nargs='+',
                        default=None)

    # evaluation
    parser.add_argument('--eval_model_path',
                        type=Path,
                        default=None)

    parser.add_argument('--nms_th',
                        type=float,
                        default=0.5)

    return parser.parse_args()


# For model evaluation.
MODEL_SPEC = [
    # DETR
    'dataset', 'model', 'cls_loss',
    'backbone', 'layer_index','n_encoder_layers', 'n_decoder_layers',
    'n_query', 'd_model', 'image_size', 'n_heads', 'd_ff', 'pos_embedding',
    'activation', 'return_intermediate', 'p_drop', 'temperature',
    'encoder_position_mode', 'decoder_sa_position_mode', 

    # Conditional DETR 
    'decoder_ca_position_mode', 'query_scale_mode', 

    # DAB-DETR
    'transformer_activation', 'num_pattern', 'modulate_wh_attn', 'iter_update',

    # DN-DETR, DINO-DETR
    'num_group', 'box_noise_scale', 'label_noise_scale',
    
    # Two-stage (DINO-DETR, Co-DETR)
    'two_stage_mode', 'two_stage_share_head', 'num_encoder_query',
]


def get_arguments():
    """Get arguments."""
    args = build_parser()
    args.save_root.mkdir(exist_ok=True, parents=True)
    log_path = args.save_root / 'logs'
    log_path.mkdir(exist_ok=True)
    
    if args.mode == 'train':
        set_random_seed(args.seed, True)
        save_dict(args.save_root / 'argument.pickle', args)
    else:
        # load args from the target experiment dir
        # and update args for load model weights.
        dir_path = args.eval_model_path.parent.parent
        arg_path = dir_path / 'argument.pickle'
        loaded_args = load_dict(arg_path)
        args = copy_target_args(loaded_args, args, MODEL_SPEC)
        args.image_set = 'test'
    return args


def copy_target_args(src, dst, targets):
    """Copy target arguments.

    dst.arg1 = src.arg1
    """
    # namespace to dict
    _src = vars(src)
    _dst = vars(dst)
    for target in targets:
        if target in _src:
            _dst[target] = _src[target]
    return argparse.Namespace(**_dst)
