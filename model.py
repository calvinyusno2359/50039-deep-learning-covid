import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Image_Dataset_Part
from torch.utils.data import DataLoader


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


if __name__ == "__main__":
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=7),
        # 32 filters in and out, no max pooling so the shapes can be added
        ResNet(
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
            )
        ),
        # Another ResNet block, you could make more of them
        # Downsampling using maxpool and others could be done in between etc. etc.
        ResNet(
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
            )
        ),
        # Pool all the 32 filters to 1, you may need to use `torch.squeeze after this layer`
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        # 32 10 classes
        torch.nn.Linear(32, 2),
    )

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
    ld_train = Image_Dataset_Part('train', img_size, class_dict, groups, dataset_numbers, dataset_paths)
    train_loader = DataLoader(ld_train, batch_size=bs_val, shuffle=True)

    # Try model on one mini-batch
    for batch_idx, (images_data, target_labels) in enumerate(train_loader):
        predicted_labels = model(images_data)
        print(predicted_labels)
        print(target_labels)
        # Forced stop
        break
        # assert False, "Forced stop after one iteration of the mini-batch for loop"
