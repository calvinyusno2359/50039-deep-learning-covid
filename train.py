import torch
import numpy as np
import torch.nn.functional as F

from datetime import datetime
from model import ResNet

from torch import optim
from torchvision import models, transforms
from dataset import TrinaryClassDataset, BinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset


def transform(img_tensor):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4824], std=[0.2363]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomCrop(150)
    ])

    return transform(img_tensor)


def validate(model, validloader, weight, epoch, lowest_loss, savePath, device='cuda'):
    model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    if weight is not None:
        weight = weight.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validloader):
            target = np.argmax(target, axis=1)
            data, target = data.to(device), target.to(device)
            data = transform(data)
            output = model(data)
            # loss = F.nll_loss(output, target, weight=weight.to(device), reduction='sum') #trinary
            test_loss += F.cross_entropy(output, target, weight=weight).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(validloader.dataset)

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validloader.dataset),
        100. * correct / len(validloader.dataset)))

    if test_loss <= lowest_loss:
        lowest_loss = test_loss
        print(f'Found New Minima at epoch {epoch} loss: {lowest_loss}\n')
        torch.save(model.state_dict(), f'{savePath}_{epoch}')

    return lowest_loss


def train(model, trainloader, weight, epoch, device='cuda'):
    print(f'Train Epoch: {epoch}')
    model.to(device)
    model.train()
    if weight is not None:
        weight = weight.to(device)
    for batch_idx, (data, target) in enumerate(trainloader):
        target = np.argmax(target, axis=1)
        data, target = data.to(device), target.to(device)
        data = transform(data)
        optimizer = optim.Adadelta(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target, weight=weight.to(device), reduction='sum') #trinary
        loss = F.cross_entropy(output, target, weight=weight, reduction='sum')
        loss.backward()
        optimizer.step()
        # TODO Change Batch size printing
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))


def train_binary_covid_clf(trainingEpochs, trainingBatchSize, savePath):
    # covid vs non-covid clf
    weight = torch.tensor([1., 1.15]) # best [1., 1.15]
    img_size = (150, 150)
    class_dict = {0: 'non-covid', 1: 'covid'}
    train_groups = ['train']
    train_numbers = {'train_non-covid': 2530,
                     'train_covid': 1345
                     }

    trainset_paths = {'train_non-covid': './dataset/train/infected/non-covid',
                      'train_covid': './dataset/train/infected/covid'
                      }

    trainset1 = BinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)

    train_numbers = {'train_non-covid': 0,
                     'train_covid': 1345 # oversample the minority
                     }

    trainset_paths = {'train_non-covid': './dataset/train/infected/non-covid',
                      'train_covid': './dataset/train/infected/covid'
                      }

    trainset2 = BinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)

    # load dataset
    trainsets = ConcatDataset([trainset1, trainset2])
    trainloader = DataLoader(trainsets, batch_size=trainingBatchSize, shuffle=True)

    val_groups = ['val']
    val_numbers = {'val_non-covid': 8,
                   'val_covid': 8,
                   }

    valset_paths = {'val_non-covid': './dataset/val/infected/non-covid',
                    'val_covid': './dataset/val/infected/covid',
                    }

    valset = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    validationloader = DataLoader(valset, batch_size=trainingBatchSize, shuffle=True)

    model = ResNet(2)

    lowest_loss = 9999
    for epoch in range(1, trainingEpochs + 1):
        train(model, trainloader, weight, epoch)
        lowest_loss = validate(model, validationloader, weight, epoch, lowest_loss, savePath)


def train_binary_normal_clf(trainingEpochs, trainingBatchSize, savePath):
    weight = torch.tensor([1., 1.2])  # best sensitivity
    img_size = (150, 150)
    class_dict = {0: 'normal', 1: 'infected'}
    groups = ['train']
    dataset_numbers = {'train_normal': 1341, # oversample the minority
                       'train_infected': 2530,
                       }

    dataset_paths = {'train_normal': './dataset/train/normal/',
                     'train_infected': './dataset/train/infected/non-covid',
                     }

    trainset1 = BinaryClassDataset('train', img_size, class_dict, groups, dataset_numbers, dataset_paths)

    dataset_numbers = {'train_normal': 1341,
                       'train_infected': 1345,
                       }

    dataset_paths = {'train_normal': './dataset/train/normal/',
                     'train_infected': './dataset/train/infected/covid',
                     }

    trainset2 = BinaryClassDataset('train', img_size, class_dict, groups, dataset_numbers, dataset_paths)

    # load dataset
    trainsets = ConcatDataset([trainset1, trainset2])
    trainloader = DataLoader(trainsets, batch_size=trainingBatchSize, shuffle=True)

    val_groups = ['val']
    val_numbers = {'val_normal': 8,
                   'val_infected': 8,
                   }

    valset_paths = {'val_normal': './dataset/test/normal',
                    'val_infected': './dataset/test/infected/covid',
                    }

    valset1 = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    valset_paths = {'val_normal': './dataset/val/normal',
                    'val_infected': './dataset/val/infected/covid',
                    }

    valset2 = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    # load dataset
    valsets = ConcatDataset([valset1, valset2])
    validationloader = DataLoader(valsets, batch_size=trainingBatchSize, shuffle=True)

    model = ResNet(2)

    lowest_loss = 9999

    for epoch in range(1, trainingEpochs + 1):
        train(model, trainloader, weight, epoch)
        lowest_loss = validate(model, validationloader, weight, epoch, lowest_loss, savePath)


def train_trinary_clf(trainingEpochs, trainingBatchSize, savePath):
    img_size = (150, 150)
    class_dict = {0: 'normal', 1: 'infected', 2: 'covid'}
    train_groups = ['train']
    train_numbers = {'train_normal': 1341,
                     'train_infected': 2530,
                     'train_covid': 1345
                     }

    trainset_paths = {'train_normal': './dataset/train/normal',
                      'train_infected': './dataset/train/infected/non-covid',
                      'train_covid': './dataset/train/infected/covid'
                      }

    trainset = TrinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)

    val_groups = ['val']
    val_numbers = {'val_normal': 234,
                   'val_infected': 242,
                   'val_covid': 139,
                   }

    valset_paths = {'val_normal': './dataset/test/normal/',
                    'val_infected': './dataset/test/infected/non-covid',
                    'val_covid': './dataset/test/infected/covid',
                    }

    valset = TrinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    # load dataset
    trainloader = DataLoader(trainset, batch_size=trainingBatchSize, shuffle=True)
    validationloader = DataLoader(valset, batch_size=trainingBatchSize, shuffle=True)

    model = ResNet(3)

    lowest_loss = 9999

    for epoch in range(1, trainingEpochs + 1):
        train(model, trainloader, weight, epoch)
        lowest_loss = validate(model, validationloader, weight, epoch, lowest_loss, savePath)


if __name__ == "__main__":
    now = datetime.now()
    timestamp = now.strftime("%d%m_%H%M")

    normalTrainingEpochs = 12
    covidTrainingEpochs = 12
    trainingBatchSize = 8
    covidSavePath = f'models/binaryModelCovid{timestamp}'
    normalSavePath = f'models/binaryModelNormal{timestamp}'
    # trinarySavePath = f'models/trinaryModel{timestamp}'

    # train_binary_normal_clf(normalTrainingEpochs, trainingBatchSize, normalSavePath)
    train_binary_covid_clf(covidTrainingEpochs, trainingBatchSize, covidSavePath)

        # train_trinary_clf(trainingEpochs, trainingBatchSize, trinarySavePath)
