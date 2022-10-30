import albumentations
import albumentations.pytorch
import numpy as np
from PIL import Image

def get_augmentations_list_from_dict(augs: dict):
    ops = []
    keys = augs.keys()
    for aug in keys:
        try:
            op = getattr(albumentations, aug)
        except:
            print(f'No aug named {aug}')  # TODO
            continue
        if issubclass(op, albumentations.BaseCompose):
            sub_augs = augs[aug]
            p = None
            if 'P' in sub_augs:
                p = sub_augs['P']
                sub_augs.pop('P')
            sub_augs_list = get_augmentations_list_from_dict(sub_augs)
            if p is None:
                op = op(sub_augs_list)
            else:
                op = op(sub_augs_list, p)
        else:
            params = tuple(augs[aug]) if augs[aug] is not None else []
            op = op(*params)
        ops.append(op)
    return ops


def get_transformations(
    mode,
    dataset_cfg,
):
    if mode == 'train':
        aug_cfg = dataset_cfg['TRAIN_AUGMENTATIONS']
    else:
        aug_cfg = dataset_cfg['TEST_AUGMENTATIONS']
    ops = get_augmentations_list_from_dict(aug_cfg)
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

