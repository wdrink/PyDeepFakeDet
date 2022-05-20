import torch.nn as nn
import torch.nn.functional as F

from PyDeepFakeDet.utils.registries import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Meso4(nn.Module):
    def __init__(self, model_cfg):
        super(Meso4, self).__init__()
        num_classes = model_cfg["NUM_CLASSES"] 
        self.conv1 = Meso1(3, 8, 3, 2)
        self.conv2 = Meso1(8, 8, 5, 2)
        self.conv3 = Meso1(8, 16, 5, 2)
        self.conv4 = Meso1(16, 16, 5, 4)
        self.fc1 = nn.Sequential(
            nn.Dropout2d(0.5), nn.Linear(16 * 8 * 8, 16), nn.LeakyReLU(0.1)
        )
        self.fc2 = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(16, num_classes))

    def forward(self, samples):
        x = samples['img']
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        output = {"logits": x}
        return output


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
