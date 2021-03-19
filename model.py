import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TrinaryClassDataset
from torch.utils.data import DataLoader

import torch.nn as nn

def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_classes=2):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1]
        ]

        self.stem = conv3x3(1, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x


# ___________________________________________ Xception ___________________________________________ #
class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()
        # Entry Flow
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.entry_2 = nn.Sequential(
            SeperableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            SeperableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_2_res = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.entry_3 = nn.Sequential(
            nn.ReLU(True),
            SeperableConv(128, 256),
            nn.BatchNorm2d(256),

            nn.ReLU(True),
            SeperableConv(256, 256),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_3_res = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)

        self.entry_4 = nn.Sequential(
            nn.ReLU(True),
            SeperableConv(256, 728),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_4_res = nn.Conv2d(256, 728, kernel_size=1, stride=2, padding=0)

        # Middle Flow
        self.middle = nn.Sequential(
            nn.ReLU(True),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728)
        )

        # Exit Flow
        self.exit_1 = nn.Sequential(
            nn.ReLU(True),
            SeperableConv(728, 728),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            SeperableConv(728, 1024),
            nn.BatchNorm2d(1024),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.exit_1_res = nn.Conv2d(728, 1024, kernel_size=1, stride=2, padding=0)
        self.exit_2 = nn.Sequential(
            SeperableConv(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),

            SeperableConv(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        stem_out = self.stem(x)
        entry_out1 = self.entry_2(stem_out) + self.entry_2_res(stem_out)
        entry_out2 = self.entry_3(entry_out1) + self.entry_3_res(entry_out1)
        entry_out3 = self.entry_4(entry_out2) + self.entry_4_res(entry_out2)

        middle_out = self.middle(entry_out3) + entry_out3

        for i in range(7):
            middle_out = self.middle(middle_out) + middle_out

        exit_out1 = self.exit_1(middle_out) + self.exit_1_res(middle_out)
        exit_out2 = self.exit_2(exit_out1)

        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.linear(exit_avg_pool_flat)

        return output


class SeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# ___________________________________________ ResNet ___________________________________________ #
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # PrintLayer(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # PrintLayer(),
            # layer1
            ResBlock(64, 64 ),
            ResBlock(64, 64),
            # layer 2
            ResBlock(64, 128, 2),
            ResBlock(128, 128),
            # layer 3
            # ResBlock(128, 256, 2),
            # ResBlock(256, 256),
            # # layer 4
            # ResBlock(256, 512, 2),
            # ResBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            # PrintLayer(),
            nn.Flatten(),
            # PrintLayer(),
        )
        # self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layers(x)
        # x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # print(stride)
        self.downsample=None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = x
        # print(res)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            res = self.downsample(res)
        # print(res.shape)
        x += res
        x = self.relu(x)
        return x
# ___________________________________________ Scrappy Residual Net ___________________________________________ #


class SimpleResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )

    def forward(self, inputs):
        return self.module(inputs) + inputs


# ___________________________________________ Helper Functions ___________________________________________ #
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


if __name__ == "__main__":
    # Create model

    # for 2 class
    model = Net(2)

    img_size = (150, 150)
    class_dict = {0: 'normal', 1: 'infected'}
    groups = ['train']
    dataset_numbers = {'train_normal': 36,
                       'train_infected': 34,
                       }

    dataset_paths = {'train_normal': './dataset_demo/train/normal/',
                     'train_infected': './dataset_demo/train/infected/',
                     }

    bs_val = 4
    ld_train = TrinaryClassDataset('train', img_size, class_dict, groups, dataset_numbers, dataset_paths)
    train_loader = DataLoader(ld_train, batch_size=bs_val, shuffle=True)

    # Try model on one mini-batch
    for batch_idx, (images_data, target_labels) in enumerate(train_loader):
        predicted_labels = model(images_data)
        print(predicted_labels)
        print(target_labels)
        # Forced stop
        break
        # assert False, "Forced stop after one iteration of the mini-batch for loop"
