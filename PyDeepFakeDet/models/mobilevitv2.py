from .modules.mobilevit_block import *
from typing import Optional, Tuple, Union, Dict
from PyDeepFakeDet.utils.registries import MODEL_REGISTRY
from .modules.mobilevitv2_block import *


@MODEL_REGISTRY.register()
class MobileViTv2(nn.Module):
    def __init__(self, model_cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        image_channels = model_cfg['layer0']['img_channels']
        out_channels = model_cfg['layer0']['out_channels']
        num_classes = model_cfg['NUM_CLASSES']

        self.model_conf_dict = dict()

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer4"]
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer5"]
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name='global_pool', module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        # self.last_pool = nn.AdaptiveAvgPool2d(1)
        # self.last_c = nn.Linear(in_features=out_channels, out_features=num_classes, bias=True)
        self.classifier.add_module(name='fc',
                                   module=nn.Linear(in_features=out_channels, out_features=num_classes, bias=True))

        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channel = cfg.get('out_channels')
        num_blocks = cfg.get('num_blocks', 2)
        expand_ratio = cfg.get('expand_ratio', 2)
        block = []
        for i in range(num_blocks):
            stride = cfg.get('stride', 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channel,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channel
        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        stride = cfg.get('stride', 1)
        output_channel = cfg.get('out_channels')
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channel,
                stride=stride,
                expand_ratio=cfg.get('mv_expand_ratio', 2)
            )

            block.append(layer)
            input_channel = output_channel

        attn_unit_dim = cfg.get("attn_unit_dim")
        ffn_multiplier = cfg.get("ffn_multiplier", 2)

        dropout = cfg.get('dropout', 0.0)

        block.append(
            MobileViTBlockv2(
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=cfg.get("ffn_dropout", 0.0),
                attn_dropout=cfg.get("attn_dropout", 0.1),
                conv_ksize=3,
                attn_norm_layer=cfg.get("attn_norm_layer", "layer_norm_2d"),
            )
        )

        return nn.Sequential(*block), input_channel



    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, samples: Dict) -> Tensor:
        x = samples['img']
        layers = {}
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.classifier(x)
        layers['logits'] = x
        return layers