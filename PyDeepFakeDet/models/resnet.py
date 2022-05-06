import torch
import torch.nn as nn

from PyDeepFakeDet.utils.registries import MODEL_REGISTRY

'''
MODEL:
  MODEL_NAME: ResNet50
  PRETRAINED: True
'''
REPO_OR_DIR = 'pytorch/vision:v0.10.0'


class BaseResNet(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            torch.hub.load(
                REPO_OR_DIR, self.net_name, pretrained=model_cfg['PRETRAINED']
            ),
            nn.Linear(1000, model_cfg['NUM_CLASSES']),
        )

    def forward(self, samples):
        x = samples['img']
        return {'logits': self.net(x)}


@MODEL_REGISTRY.register()
class ResNet18(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet18'
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet34(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet34'
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet50(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet50'
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet101(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet101'
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet152(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet152'
        super().__init__(model_cfg)
