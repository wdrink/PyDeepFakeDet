import random

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyDeepFakeDet.utils.registries import MODEL_REGISTRY

from .efficientnet import EfficientNet


class AGDA(nn.Module):
    def __init__(self, AGDA_cfg):
        super().__init__()
        kernel_size = AGDA_cfg['AGDA_KERNEL_SIZE']  # 7
        dilation = AGDA_cfg['AGDA_DILATION']  # 2
        sigma = AGDA_cfg['AGDA_SIGMA']  # 5,
        threshold = AGDA_cfg['AGDA_THRESHOLD']  # (0.4,0.6),
        zoom = AGDA_cfg['AGDA_ZOOM']  # (3,5),
        scale_factor = AGDA_cfg['AGDA_SCALE_FACTOR']  # 0.5,
        noise_rate = AGDA_cfg['AGDA_NOISE_RATE']  # 0.1,
        mode = AGDA_cfg['AGDA_MODE']  # 'soft'
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = sigma
        self.noise_rate = noise_rate
        self.scale_factor = scale_factor
        self.threshold = tuple(threshold)
        self.zoom = tuple(zoom)
        self.mode = mode
        self.filter = kornia.filters.GaussianBlur2d(
            (self.kernel_size, self.kernel_size), (self.sigma, self.sigma)
        )
        if mode == 'pointwise':
            distmap = (np.arange(kernel_size) - (kernel_size - 1) / 2) ** 2
            distmap = distmap.reshape(1, -1) + distmap.reshape(-1, 1)
            distmap *= dilation ** 2
            rec = {}
            for i in range(kernel_size):
                for j in range(kernel_size):
                    d = distmap[i, j]
                    if d not in rec:
                        rec[d] = [(i, j)]
                    else:
                        rec[d].append((i, j))
            dist = np.array(list(rec.keys()))
            mod = np.zeros([dist.shape[0], kernel_size, kernel_size])
            ct = []
            for i in range(dist.shape[0]):
                ct.append(len(rec[dist[i]]))
                for j in rec[dist[i]]:
                    mod[i, j[0], j[1]] = 1
            mod = (
                torch.from_numpy(mod)
                .reshape(-1, 1, kernel_size, kernel_size)
                .repeat([3, 1, 1, 1])
                .float()
            )
            self.register_buffer('kernel', mod)
            self.register_buffer(
                'dist', torch.from_numpy(dist).reshape(1, -1, 1, 1).float()
            )
            self.register_buffer(
                'ct', torch.Tensor(ct).reshape(1, -1, 1, 1).float()
            )
        self.AGDA_cfg = AGDA_cfg
        self.AGDA_loss_weight = AGDA_cfg['AGDA_LOSS_WEIGHT']

    def soft_drop(self, X, attention_map):
        with torch.no_grad():
            B, C, H, W = X.size()
            X = F.pad(
                X,
                tuple([int((self.kernel_size - 1) / 2 * self.dilation)] * 4),
                mode='reflect',
            )
            X = F.conv2d(X, self.kernel, dilation=self.dilation, groups=3).view(
                B, 3, -1, H, W
            )
            sigma = attention_map * self.sigma
            ex = torch.exp(-1.0 / (2 * sigma)).view(B, 1, H, W)
            ex = ex ** self.dist
            c = torch.sum(self.ct * ex, dim=1, keepdim=True)
            X = torch.einsum('bijmn,bjmn->bimn', X, ex)
            X /= c
        return X

    def mod_func(self, x):
        if type(self.threshold) == tuple:
            thres = random.uniform(*self.threshold)
        else:
            thres = self.threshold
        if type(self.zoom) == tuple:
            zoom = random.uniform(*self.zoom)
        else:
            zoom = self.zoom
        bottom = torch.sigmoid((torch.tensor(0.0) - thres) * zoom)
        return (torch.sigmoid((x - thres) * zoom) - bottom) / (1 - bottom)

    def soft_drop2(self, x, attention_map):
        with torch.no_grad():
            attention_map = self.mod_func(attention_map)
            B, C, H, W = x.size()
            xs = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=True,
            )
            xs = self.filter(xs)
            xs += torch.randn_like(xs) * self.noise_rate
            xs = F.interpolate(xs, (H, W), mode='bilinear', align_corners=True)
            x = x * (1 - attention_map) + xs * attention_map
        return x

    def hard_drop(self, X, attention_map):
        with torch.no_grad():
            if type(self.threshold) == tuple:
                thres = random.uniform(*self.threshold)
            else:
                thres = self.threshold
            attention_mask = attention_map < thres
            X = attention_mask.float() * X
        return X

    def mix_drop(self, X, attention_map):
        if random.randint(0, 1) == 0:
            return self.hard_drop(X, attention_map)
        else:
            return self.soft_drop2(X, attention_map)

    def agda(self, X, attention_map):
        with torch.no_grad():
            attention_weight = torch.sum(attention_map, dim=(2, 3))
            attention_map = F.interpolate(
                attention_map,
                (X.size(2), X.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            attention_weight = torch.sqrt(attention_weight + 1)
            index = torch.distributions.categorical.Categorical(
                attention_weight
            ).sample()
            index1 = index.view(-1, 1, 1, 1).repeat(1, 1, X.size(2), X.size(3))
            attention_map = torch.gather(attention_map, 1, index1)
            atten_max = (
                torch.max(attention_map.view(attention_map.shape[0], 1, -1), 2)[
                    0
                ]
                + 1e-8
            )
            attention_map = attention_map / atten_max.view(
                attention_map.shape[0], 1, 1, 1
            )
            if self.mode == 'soft':
                return self.soft_drop2(X, attention_map), index
            elif self.mode == 'pointwise':
                return self.soft_drop(X, attention_map), index
            elif self.mode == 'hard':
                return self.hard_drop(X, attention_map), index
            elif self.mode == 'mix':
                return self.mix_drop(X, attention_map), index


class AttentionMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels
        self.conv_extract = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1
        )  # extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(
            self.mask, (x.shape[2], x.shape[3]), mode='nearest'
        )
        return x * mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(
                attentions, size=(H, W), mode='bilinear', align_corners=True
            )
        if norm == 1:
            attentions = attentions + 1e-8
        if len(features.shape) == 4:
            feature_matrix = torch.einsum(
                'imjk,injk->imn', attentions, features
            )
        else:
            feature_matrix = torch.einsum(
                'imjk,imnjk->imn', attentions, features
            )
        if norm == 1:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        if norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm == 3:
            w = torch.sum(attentions, dim=(2, 3)).unsqueeze(-1) + 1e-8
            feature_matrix /= w
        return feature_matrix


class Texture_Enhance(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features
        self.output_features_d = num_features
        self.conv_extract = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv0 = nn.Conv2d(
            num_features * num_attentions,
            num_features * num_attentions,
            5,
            padding=2,
            groups=num_attentions,
        )
        self.conv1 = nn.Conv2d(
            num_features * num_attentions,
            num_features * num_attentions,
            3,
            padding=1,
            groups=num_attentions,
        )
        self.bn1 = nn.BatchNorm2d(num_features * num_attentions)
        self.conv2 = nn.Conv2d(
            num_features * 2 * num_attentions,
            num_features * num_attentions,
            3,
            padding=1,
            groups=num_attentions,
        )
        self.bn2 = nn.BatchNorm2d(2 * num_features * num_attentions)
        self.conv3 = nn.Conv2d(
            num_features * 3 * num_attentions,
            num_features * num_attentions,
            3,
            padding=1,
            groups=num_attentions,
        )
        self.bn3 = nn.BatchNorm2d(3 * num_features * num_attentions)
        self.conv_last = nn.Conv2d(
            num_features * 4 * num_attentions,
            num_features * num_attentions,
            1,
            groups=num_attentions,
        )
        self.bn4 = nn.BatchNorm2d(4 * num_features * num_attentions)
        self.bn_last = nn.BatchNorm2d(num_features * num_attentions)

        self.M = num_attentions

    def cat(self, a, b):
        B, C, H, W = a.shape
        c = torch.cat(
            [a.reshape(B, self.M, -1, H, W), b.reshape(B, self.M, -1, H, W)],
            dim=2,
        ).reshape(B, -1, H, W)
        return c

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (
                int(H * attention_maps[0]),
                int(W * attention_maps[1]),
            )
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps = self.conv_extract(feature_maps)
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        if feature_maps.size(2) > feature_maps_d.size(2):
            feature_maps = feature_maps - F.interpolate(
                feature_maps_d,
                (feature_maps.shape[2], feature_maps.shape[3]),
                mode='nearest',
            )
        attention_maps = (
            (
                torch.tanh(
                    F.interpolate(
                        attention_maps.detach(),
                        (H, W),
                        mode='bilinear',
                        align_corners=True,
                    )
                )
            ).unsqueeze(2)
            if type(attention_maps) != tuple
            else 1
        )
        feature_maps = feature_maps.unsqueeze(1)
        feature_maps = (feature_maps * attention_maps).reshape(B, -1, H, W)
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(
            F.relu(self.bn1(feature_maps0), inplace=True)
        )
        feature_maps1_ = self.cat(feature_maps0, feature_maps1)
        feature_maps2 = self.conv2(
            F.relu(self.bn2(feature_maps1_), inplace=True)
        )
        feature_maps2_ = self.cat(feature_maps1_, feature_maps2)
        feature_maps3 = self.conv3(
            F.relu(self.bn3(feature_maps2_), inplace=True)
        )
        feature_maps3_ = self.cat(feature_maps2_, feature_maps3)
        feature_maps = F.relu(
            self.bn_last(
                self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True))
            ),
            inplace=True,
        )
        feature_maps = feature_maps.reshape(B, -1, N, H, W)
        return feature_maps, feature_maps_d


@MODEL_REGISTRY.register()
class MAT(nn.Module):
    def __init__(self, model_config):
        super(MAT, self).__init__()
        net = model_config['NET']  # 'xception',
        feature_layer = model_config['FEATURE_LAYER']  # 'b3',
        attention_layer = model_config['ATTENTION_LAYER']  # 'final',
        num_classes = 2
        num_attentions = model_config['NUM_ATTENTIONS']  # 8,
        mid_dims = model_config['MID_DIMS']  # 256,
        dropout_rate = model_config['DROPOUT_RATE']  # 0.5,
        drop_final_rate = model_config['DROP_FINAL_RATE']  # 0.5,
        alpha = model_config['ALPHA']  # 0.05,
        size = model_config['SIZE']  # (380, 380),
        margin = model_config['MARGIN']  # 1,
        inner_margin = model_config['INNER_MARGIN']  # [0.01, 0.02],
        self.num_classes = num_classes
        self.M = num_attentions
        self.net = EfficientNet({'VARIANT': net, 'PRETRAINED': True})
        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        with torch.no_grad():
            layers = self.net({'img': torch.zeros(1, 3, size[0], size[1])})
        num_features = layers[self.feature_layer].shape[1]
        self.mid_dims = mid_dims
        self.attentions = AttentionMap(
            layers[self.attention_layer].shape[1], self.M
        )
        self.atp = AttentionPooling()
        self.texture_enhance = Texture_Enhance(num_features, num_attentions)
        self.num_features = self.texture_enhance.output_features
        self.num_features_d = self.texture_enhance.output_features_d
        self.projection_local = nn.Sequential(
            nn.Linear(num_attentions * self.num_features, mid_dims),
            nn.Hardswish(),
            nn.Linear(mid_dims, mid_dims),
        )
        self.project_final = nn.Linear(layers['final'].shape[1], mid_dims)
        # print(mid_dims,num_classes)
        self.ensemble_classifier_fc = nn.Sequential(
            nn.Linear(mid_dims * 2, mid_dims),
            nn.Hardswish(),
            nn.Linear(mid_dims, num_classes),
        )
        loss_cfg = {
            'AUX_LOSS_WEIGHT': 1,
            'M': num_attentions,
            'N': self.num_features_d,
            'C': num_classes,
            'ALPHA': alpha,
            'MARGIN': margin,
            'INNER_MARGIN': inner_margin,
        }
        from PyDeepFakeDet.utils.loss import Auxiliary_Loss_v2

        self.auxiliary_loss = Auxiliary_Loss_v2(loss_cfg)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_final = nn.Dropout(drop_final_rate, inplace=True)

        if model_config['AGDA_CONFIG']['AGDA_LOSS_WEIGHT'] != 0:
            self.AG = AGDA(model_config['AGDA_CONFIG'])
        else:
            self.AG = None

    def train_batch(self, x, y, drop_final=False):
        y = torch.squeeze(y).long()
        layers = self.net({'img': x})
        if self.feature_layer == 'logits':
            logits = layers['logits']
            loss = F.cross_entropy(logits, y)
            return dict(loss=loss, logits=logits)
        feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_ = self.attentions(raw_attentions)
        dropout_mask = self.dropout(
            torch.ones([attention_maps_.shape[0], self.M, 1], device=x.device)
        )
        attention_maps = attention_maps_ * torch.unsqueeze(dropout_mask, -1)
        feature_maps, feature_maps_d = self.texture_enhance(
            feature_maps, attention_maps_
        )
        feature_maps_d = feature_maps_d - feature_maps_d.mean(
            dim=[2, 3], keepdim=True
        )
        feature_maps_d = feature_maps_d / (
            torch.std(feature_maps_d, dim=[2, 3], keepdim=True) + 1e-8
        )
        feature_matrix_ = self.atp(feature_maps, attention_maps_)
        feature_matrix = feature_matrix_ * dropout_mask

        B, M, N = feature_matrix.size()
        aux_loss, feature_matrix_d = self.auxiliary_loss(
            feature_maps_d, attention_maps_, y
        )
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps, norm=1).squeeze(1)
        final = self.dropout_final(final)
        projected_final = F.hardswish(self.project_final(final))
        if drop_final:
            projected_final *= 0
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        ensemble_loss = F.cross_entropy(ensemble_logit, y)
        return dict(
            ensemble_loss=ensemble_loss,
            aux_loss=aux_loss,
            attention_maps=attention_maps_,
            ensemble_logit=ensemble_logit,
            feature_matrix=feature_matrix_,
            feature_matrix_d=feature_matrix_d,
        )

    def forward(self, samples):
        x = samples['img']
        y = samples['bin_label']
        if self.training:
            if self.AG is None:
                return self.train_batch(x, y)
            else:
                loss_pack = self.train_batch(x, y)
                with torch.no_grad():
                    Xaug, index = self.AG.agda(x, loss_pack['attention_maps'])
                loss_pack2 = self.train_batch(Xaug, y)
                loss_pack['AGDA_ensemble_loss'] = loss_pack2['ensemble_loss']
                loss_pack['AGDA_aux_loss'] = loss_pack2['aux_loss']
                one_hot = F.one_hot(index, self.M)
                loss_pack['match_loss'] = torch.mean(
                    torch.norm(
                        loss_pack2['feature_matrix_d']
                        - loss_pack['feature_matrix_d'],
                        dim=-1,
                    )
                    * (torch.ones_like(one_hot) - one_hot)
                )
                return loss_pack
        layers = self.net({'img': x})
        if self.feature_layer == 'logits':
            layers.update({'loss': torch.tensor(0.0)})
            return layers
        raw_attentions = layers[self.attention_layer]
        attention_maps = self.attentions(raw_attentions)
        feature_maps = layers[self.feature_layer]
        feature_maps, feature_maps_d = self.texture_enhance(
            feature_maps, attention_maps
        )
        feature_matrix = self.atp(feature_maps, attention_maps)
        B, M, N = feature_matrix.size()
        feature_matrix = self.dropout(feature_matrix)
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
        final = self.atp(final, attention_maps2, norm=1).squeeze(1)
        projected_final = F.hardswish(self.project_final(final))
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        return dict(logits=ensemble_logit, loss=torch.tensor(0.0))
