import torch
import torch.nn as nn
import torch.nn.functional as F

from PyDeepFakeDet.utils.registries import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MesoInception4(nn.Module):
    def __init__(self, model_cfg):
        super(MesoInception4, self).__init__()
        num_classes = model_cfg["NUM_CLASSES"]
        self.inception_conv1 = InceptionLayer(3, 1, 4, 4, 2)
        self.inception_conv2 = InceptionLayer(11, 2, 4, 4, 2)
        self.conv3 = Meso1(12, 16, 5, 2)
        self.conv4 = Meso1(16, 16, 5, 4)
        self.fc1 = nn.Sequential(
            nn.Dropout2d(0.5), nn.Linear(16 * 8 * 8, 16), nn.LeakyReLU(0.1)
        )
        self.fc2 = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(16, num_classes))

    def forward(self, samples):
        x = samples['img']
        x = self.inception_conv1(x)
        x = self.inception_conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        output = {"logits": x}
        return output


class InceptionLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel1,
        out_channel2,
        out_channel3,
        out_channel4,
    ):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel1, padding="same", kernel_size=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel2, 1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, 3, padding="same"),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel3, 1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(
                out_channel3, out_channel3, 3, padding="same", dilation=2
            ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel4, 1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(
                out_channel4, out_channel4, 3, padding="same", dilation=3
            ),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(
            out_channel1 + out_channel2 + out_channel3 + out_channel4
        )
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        y = torch.cat((x1, x2, x3, x4), dim=1)
        y = self.bn(y)
        y = self.pool(y)
        return y


class Meso1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(Meso1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="same", bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d((pool_size, pool_size))

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        return x
