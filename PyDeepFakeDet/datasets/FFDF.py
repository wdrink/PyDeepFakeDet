import os

import torch

from PyDeepFakeDet.datasets.dataset import DeepFakeDataset
from PyDeepFakeDet.datasets.utils import get_image_from_path
from PyDeepFakeDet.utils.registries import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FFDF(DeepFakeDataset):
    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        image_info = info_line.strip('\n').split()
        image_path = image_info[0]
        image_abs_path = os.path.join(self.root_dir, image_path)

        img = get_image_from_path(
            image_abs_path, self.mode, self.dataset_cfg
        )
        img_label_binary = int(image_info[1])

        sample = {
            'img': img,
            'bin_label': [int(img_label_binary)],
        }

        sample['img'] = torch.FloatTensor(sample['img'])
        sample['bin_label'] = torch.FloatTensor(sample['bin_label'])
        sample['bin_label_onehot'] = self.label_to_one_hot(
            sample['bin_label'], 2
        ).squeeze()

        return sample
