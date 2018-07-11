import torch
import torchvision.transforms as transforms
from transforms import resize, random_crop, random_distort, random_flip, random_paste

def init_transforms(opt, subset):

    model_name = opt.model
    collate_fn = None

    if model_name in ['fpnssd', 'ssd300', 'ssd512']:

        from modellibs.ssd.transforms import SSDAugmentation
        transform = SSDAugmentation(opt, subset)
    elif model_name in ['refinedet']:
        from modellibs.refinedet.transform import Augmentation
        transform = Augmentation(opt, subset)

    elif model_name in ['s3fd']:
        from modellibs.s3fd.transforms import S3FDAugmentation
        transform = S3FDAugmentation(opt, subset)

    elif model_name in ['resnet']:
        from modellibs.resnet.transforms import ResnetAugmentation
        transform = ResnetAugmentation(opt, subset)

    elif model_name in ['chanet']:
        from modellibs.resnet.transforms import ResnetAugmentation
        transform = ResnetAugmentation(opt, subset)

    else:
        raise ValueError('Not a valid model')

    return transform