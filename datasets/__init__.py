from .shwd import get_shwd_dataloader


def get_dataloader(args, shuffle=True, drop_last=True):
    if args.dataset == 'shwd':
        return get_shwd_dataloader(args, shuffle, drop_last)
