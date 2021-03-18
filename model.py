import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TrinaryClassDataset
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, numberOfOutputLabels=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # PrintLayer(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # PrintLayer(),
            # layer1
            ResBlock(64, 64,),
            ResBlock(64, 64),
            # layer 2
            ResBlock(64, 128, 2),
            ResBlock(128, 128),
            # # layer 3
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
        self.fc2 = nn.Linear(128, numberOfOutputLabels)

    def forward(self, x):
        x = self.layers(x)
        # x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


#
# class SimpleResBlock(nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#         )
#
#     def forward(self, inputs):
#         return self.module(inputs) + inputs

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # print(stride)
        # if stride != 1:
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # if self.downsample is not None:
        residual = self.downsample(x)
        # print(residual.shape)
        out += residual
        out = self.relu(out)

        return out


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x



if __name__ == "__main__":
    # Create model

    # for 2 class
    model = Net(numberOfOutputLabels=3)

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
