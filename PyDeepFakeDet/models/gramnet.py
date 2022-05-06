import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyDeepFakeDet.utils.registries import MODEL_REGISTRY

from .modules.gram_block import GramBlock
from .xception import Block, SeparableConv2d


@MODEL_REGISTRY.register()
class GramNet(nn.Module):
    def __init__(self, model_cfg):
        super(GramNet, self).__init__()
        num_classes = model_cfg['NUM_CLASSES']
        pretrained = model_cfg['PRETRAINED']

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # entry flow
        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True
        )
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True
        )
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True
        )

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True
        )

        # exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False
        )

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)

        # in_channel = 2048 + gram_block's out_channel * 9
        self.fc = nn.Linear(2336, num_classes)

        # gram feature for concatenation
        self.gram = None

        # gram blocks
        self.gram_block_3 = GramBlock(3)
        self.gram_block_32 = GramBlock(32)
        self.gram_block_64 = GramBlock(64)
        self.gram_block_128 = GramBlock(128)
        self.gram_block_256 = GramBlock(256)
        self.gram_block_728_1 = GramBlock(728)
        self.gram_block_728_2 = GramBlock(728)
        self.gram_block_1024 = GramBlock(1024)
        self.gram_block_1536 = GramBlock(1536)

        if pretrained == 'imagenet':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
            )
            if num_classes != 1000:
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if (k in self.state_dict() and 'fc' not in k)
                }
            self.load_state_dict(
                pretrained_dict,
                strict=False,
            )
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, samples):
        x = samples['img']
        self.gram = self.gram_block_3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        self.gram = torch.cat([self.gram, self.gram_block_32(x)], dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        self.gram = torch.cat([self.gram, self.gram_block_64(x)], dim=1)

        x = self.block1(x)
        self.gram = torch.cat([self.gram, self.gram_block_128(x)], dim=1)
        x = self.block2(x)
        self.gram = torch.cat([self.gram, self.gram_block_256(x)], dim=1)
        x = self.block3(x)
        self.gram = torch.cat([self.gram, self.gram_block_728_1(x)], dim=1)

        # middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        self.gram = torch.cat([self.gram, self.gram_block_728_2(x)], dim=1)

        # exit flow
        x = self.block12(x)
        self.gram = torch.cat([self.gram, self.gram_block_1024(x)], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        self.gram = torch.cat([self.gram, self.gram_block_1536(x)], dim=1)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.cat([self.gram, x], dim=1)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        output = {'logits': x}
        return output
