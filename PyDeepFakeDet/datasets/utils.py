import albumentations
import albumentations.pytorch
import numpy as np
from PIL import Image

def get_augmentations_from_list(augs: list, aug_cfg, one_of_p=1):
    ops = []
    for aug in augs:
        if isinstance(aug, list):
            op = albumentations.OneOf
            param = get_augmentations_from_list(aug, aug_cfg)
            param = [param, one_of_p]
        else:
            op = getattr(albumentations, aug)
            param = (
                aug_cfg[aug.upper() + '_PARAMS']
                if aug.upper() + '_PARAMS' in aug_cfg
                else []
            )
        ops.append(op(*tuple(param)))
    return ops


def get_transformations(
    mode,
    dataset_cfg,
):
    if mode == 'train':
        aug_cfg = dataset_cfg['TRAIN_AUGMENTATIONS']
    else:
        aug_cfg = dataset_cfg['TEST_AUGMENTATIONS']
    ops = get_augmentations_from_list(aug_cfg['COMPOSE'], aug_cfg)
    ops.append(albumentations.pytorch.transforms.ToTensorV2())
    augmentations = albumentations.Compose(ops, p=1)
    return augmentations


def get_image_from_path(img_path, mode, dataset_cfg):
    img = Image.open(img_path)

    trans_list = get_transformations(
        mode,
        dataset_cfg,
    )
    if mode == 'train':
        img = np.asarray(img)
        img = trans_list(image=img)['image']

    else:
        img = np.asarray(img)
        img = trans_list(image=img)['image']

    return img

