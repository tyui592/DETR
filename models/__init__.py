from .detr import get_detr, get_detr_criterion
from .conditional_detr import get_conditional_detr, get_conditional_detr_criterion
from .dab_detr import get_dab_detr, get_dab_detr_criterion
from .dn_detr import get_dn_detr, get_dn_detr_criterion
from .dino_detr import get_dino_detr, get_dino_detr_criterion
from .dino_detr_two_stage import get_dino_detr2, get_dino_detr2_criterion
from .postprocessor import CEPostProcess, FocalPostProcess

def get_model(args, device):
    if args.model == 'detr':
        model = get_detr(args, device)
        criterion = get_detr_criterion(args, device)

    elif args.model == 'conditional_detr':
        model = get_conditional_detr(args, device)
        criterion = get_conditional_detr_criterion(args, device)

    elif args.model =='dab-detr':
        model = get_dab_detr(args, device)
        criterion = get_dab_detr_criterion(args, device)

    elif args.model == 'dn-detr':
        model = get_dn_detr(args, device)
        criterion = get_dn_detr_criterion(args, device)

    elif args.model == 'dino-detr':
        model = get_dino_detr(args, device)
        criterion = get_dino_detr_criterion(args, device)

    elif args.model == 'dino-detr2':
        model = get_dino_detr2(args, device)
        criterion = get_dino_detr2_criterion(args, device)

    if args.cls_loss == 'ce':
        postprocessor = CEPostProcess()
    else:
        postprocessor = FocalPostProcess()

    return model, criterion, postprocessor
