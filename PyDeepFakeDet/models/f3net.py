import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyDeepFakeDet.utils.registries import MODEL_REGISTRY
from PyDeepFakeDet.models.xception import Xception

@MODEL_REGISTRY.register()
class F3Net(nn.Module):
    def __init__(
        self,
        model_cfg
    ):
        super(F3Net, self).__init__()
        assert model_cfg['IMG_WIDTH'] == model_cfg['IMG_HEIGHT']
        self.model_cfg = model_cfg
        img_size = model_cfg['IMG_WIDTH']
        self.num_classes = model_cfg['NUM_CLASSES']
        self.window_size = model_cfg['LFS_WINDOW_SIZE']
        self._LFS_M = model_cfg['LFS_M']

        self.FAD_head = FAD_Head(img_size)
        self.init_xcep_FAD()
        self.LFS_head = LFS_Head(img_size, self.window_size, self._LFS_M)
        self.init_xcep_LFS()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            4096,
            self.num_classes,
        )
        self.dp = nn.Dropout(p=0.2)

    def forward(self, samples):
        x = samples['img']
        fea_FAD = self.FAD_head(x)
        fea_FAD = self.FAD_xcep({"img":fea_FAD})['final']
        fea_FAD = self._norm_fea(fea_FAD)
        fea_LFS = self.LFS_head(x)
        fea_LFS = self.LFS_xcep({"img":fea_LFS})['final']
        fea_LFS = self._norm_fea(fea_LFS)
        y = torch.cat((fea_FAD, fea_LFS), dim=1)
        f = self.dp(y)
        f = self.fc(f)
        f = F.softmax(f, dim=-1)
        output = {'logits': f}
        return output

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        f = f.view(f.size(0), -1)
        return f
    
    def init_xcep_FAD(self):
        self.FAD_xcep = Xception(self.model_cfg['XCEPTION_CFG'], 12)
        

    def init_xcep_LFS(self):
        self.LFS_xcep = Xception(self.model_cfg['XCEPTION_CFG'], self._LFS_M)


class Filter(nn.Module):
    def __init__(
        self, size, band_start, band_end, norm=False
    ):
        super(Filter, self).__init__()

        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        self.learnable = nn.Parameter(
                torch.randn(size, size), requires_grad=True
            )
        self.learnable.data.normal_(0.0, 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(
                    torch.tensor(generate_filter(band_start, band_end, size))
                ),
                requires_grad=False,
            )

    def forward(self, x):
        filt = self.base + norm_sigma(self.learnable)

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()
        self._DCT_all = nn.Parameter(
            torch.tensor(DCT_mat(size)).float(), requires_grad=False
        )
        self._DCT_all_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1),
            requires_grad=False,
        )
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList(
            [low_filter, middle_filter, high_filter, all_filter]
        )

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T 
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq) 
            y = self._DCT_all_T @ x_pass @ self._DCT_all  
            y_list.append(y)
        out = torch.cat(y_list, dim=1) 
        return out


class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        self._DCT_patch = nn.Parameter(
            torch.tensor(DCT_mat(window_size)).float(), requires_grad=False
        )
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
            requires_grad=False,
        )

        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=2, padding=4
        )

        self.filters = nn.ModuleList(
            [
                Filter(
                    window_size,
                    window_size * 2.0 / M * i,
                    window_size * 2.0 / M * (i + 1),
                    norm=True,
                )
                for i in range(M)
            ]
        )

    def forward(self, x):
        x_gray = (
            0.299 * x[:, 0, :, :]
            + 0.587 * x[:, 1, :, :]
            + 0.114 * x[:, 2, :, :]
        )
        x = x_gray.unsqueeze(1)
        x = (x + 1.0) * 122.5

        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8) / 2) + 1
        assert size_after == 149

        x_unfold = self.unfold(x)  
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(
            N, L, C, S, S
        )  
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        y_list = []
        for i in range(self._M):
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2, 3, 4])
            y = y.reshape(N, size_after, size_after).unsqueeze(
                dim=1
            )  
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  
        return out


def DCT_mat(size):
    m = [
        [
            (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
            * np.cos((j + 0.5) * np.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]
    return m


def generate_filter(start, end, size):
    return [
        [0.0 if i + j > end or i + j <= start else 1.0 for j in range(size)]
        for i in range(size)
    ]


def norm_sigma(x):
    return 2.0 * torch.sigmoid(x) - 1.0
