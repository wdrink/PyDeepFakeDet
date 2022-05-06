from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyDeepFakeDet.utils.registries import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss.
    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.
    """

    def __init__(self, loss_cfg):
        super().__init__()
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT']).cuda()
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def forward(self, outputs, samples, **kwargs):
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if cls_score.size() == label.size():
            # calculate loss for soft labels
            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                lsm = lsm * self.class_weight.unsqueeze(0).cuda()
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0).cuda() * label
                )
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label
            loss_cls = F.cross_entropy(
                cls_score, label, self.class_weight, **kwargs
            )
        return loss_cls


@LOSS_REGISTRY.register()
class BCELossWithLogits(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT']).cuda()
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def forward(self, outputs, samples, **kwargs):
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_score, label, self.class_weight
        )
        return loss_cls


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, samples, **kwargs):
        pred_mask = outputs['pred_mask']
        gt_mask = samples['mask']
        loss = self.mse(pred_mask, gt_mask)
        return loss


@LOSS_REGISTRY.register()
class FocalLoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.alpha = loss_cfg['ALPHA'] if 'ALPHA' in loss_cfg.keys() else 1
        self.gamma = loss_cfg['GAMMA'] if 'GAMMA' in loss_cfg.keys() else 2
        self.logits = (
            loss_cfg['LOGITS'] if 'LOGITS' in loss_cfg.keys() else True
        )
        self.reduce = (
            loss_cfg['REDUCE'] if 'REDUCE' in loss_cfg.keys() else True
        )

    def forward(self, outputs, samples, **kwargs):
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                cls_score, label, reduction='none'
            )
        else:
            BCE_loss = F.binary_cross_entropy(
                cls_score, label, reduction='none'
            )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


@LOSS_REGISTRY.register()
class Auxiliary_Loss_v2(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        M = loss_cfg['M'] if 'M' in loss_cfg.keys() else 1
        N = loss_cfg['N'] if 'N' in loss_cfg.keys() else 1
        C = loss_cfg['C'] if 'C' in loss_cfg.keys() else 1
        alpha = loss_cfg['ALPHA'] if 'ALPHA' in loss_cfg.keys() else 0.05
        margin = loss_cfg['MARGIN'] if 'MARGIN' in loss_cfg.keys() else 1
        inner_margin = (
            loss_cfg['INNER_MARGIN']
            if 'INNER_MARGIN' in loss_cfg.keys()
            else [0.1, 5]
        )

        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        from PyDeepFakeDet.models.matdd import AttentionPooling

        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(
                attentions, (H, W), mode='bilinear', align_corners=True
            )
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers
        center_momentum = feature_matrix - feature_centers
        real_mask = (y == 0).view(-1, 1, 1)
        fcts = (
            self.alpha * torch.mean(center_momentum * real_mask, dim=0)
            + feature_centers
        )
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        fctsd, torch.distributed.ReduceOp.SUM
                    )
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = self.inner_margin[y]
        intra_class_loss = F.relu(
            torch.norm(feature_matrix - fcts, dim=[1, 2])
            * torch.sign(inner_margin)
            - inner_margin
        )
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(
                    self.margin - torch.dist(fcts[j], fcts[k]), inplace=False
                )
        inter_class_loss = inter_class_loss / M / self.alpha
        return intra_class_loss + inter_class_loss, feature_matrix


@LOSS_REGISTRY.register()
class Multi_attentional_Deepfake_Detection_loss(nn.Module):
    def __init__(self, loss_cfg) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    def forward(self, loss_pack, label):
        if 'loss' in loss_pack:
            return loss_pack['loss']
        loss = (
            self.loss_cfg['ENSEMBLE_LOSS_WEIGHT'] * loss_pack['ensemble_loss']
            + self.loss_cfg['AUX_LOSS_WEIGHT'] * loss_pack['aux_loss']
        )
        if self.loss_cfg['AGDA_LOSS_WEIGHT'] != 0:
            loss += (
                self.loss_cfg['AGDA_LOSS_WEIGHT']
                * loss_pack['AGDA_ensemble_loss']
                + self.loss_cfg['MATCH_LOSS_WEIGHT'] * loss_pack['match_loss']
            )
        return loss
