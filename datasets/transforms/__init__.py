from .albumentation_transforms import get_transforms as get_alb_transforms
from .detr_transforms import get_transforms as get_detr_tramsforms

def get_transforms(args):
    if args.aug_policy < 6:
        transforms, collate_fn = get_alb_transforms(args)
    else:
        transforms, collate_fn = get_detr_tramsforms(args)
    return transforms, collate_fn
