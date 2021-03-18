import time
import torch
import numpy as np
import torch.nn.functional as F

from model import Net

from torch import nn
from torch import optim
from torchvision import models
from dataset import TrinaryClassDataset, BinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset


def validate(model, validloader, device='cuda'):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validloader):
            target = np.argmax(target, axis=1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # predicted_labels = torch.exp(output).max(dim=1)[1]
            # equality = (target.data.max(dim=1)[1] == predicted_labels)
            # accuracy += equality.type(torch.FloatTensor).mean()
            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            # 	test_loss, correct, len(validloader.dataset),
            # 	100. * correct / len(validloader.dataset)))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(validloader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validloader.dataset),
        100. * correct / len(validloader.dataset)))


# return test_loss, accuracy


def train(model, trainloader, epoch, device='cuda'):
    print(f'Train Epoch: {epoch}')
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # print("TARGET", target)
        target = np.argmax(target, axis=1)
        # print("TARGET", target)
        # print("DATA", data)
        data, target = data.to(device), target.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0002)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # TODO Change Batch size printing
        if batch_idx % 250 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))

def train_multi_clf():
    # set and load dataset spec
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
    batch_size = 4
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    epochs = 10
    model = Net()

    for epoch in range(1, epochs + 1):
        train(model, trainloader, epoch)
        validate(model, validationloader)


def train_binary_covid_clf(trainingEpochs, trainingBatchSize, savePath):
	# covid vs non-covid clf
	img_size = (150, 150)
	class_dict = {0: 'non-covid', 1: 'covid'}
	train_groups = ['train']
	train_numbers = {'train_non-covid': 2530,
					 'train_covid': 1345
					 }

	trainset_paths = {'train_non-covid': './dataset/train/infected/non-covid',
	                  'train_covid': './dataset/train/infected/covid'
	                  }

	trainset = BinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)

	val_groups = ['val']
	val_numbers = {'val_non-covid': 8,
	               'val_covid': 8,
	               }

	valset_paths = {'val_non-covid': './dataset/val/infected/non-covid',
	                'val_covid': './dataset/val/infected/covid',
	                }

	valset = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

	# load dataset
	trainloader = DataLoader(trainset, batch_size=trainingBatchSize, shuffle=True)
	validationloader = DataLoader(valset, batch_size=trainingBatchSize, shuffle=True)

	model = Net(numberOfOutputLabels=2)

	for epoch in range(1, trainingEpochs + 1):
		train(model, trainloader, epoch)
		validate(model, validationloader)
		torch.save(model.state_dict(), savePath)


def train_covid_clf():
    # covid vs non-covid clf
    img_size = (150, 150)
    class_dict = {0: 'non-covid', 1: 'covid'}
    train_groups = ['train']
    train_numbers = {'train_non-covid': 2530,
                     'train_covid': 1345,
                     }
    #
    trainset_paths = {'train_non-covid': './dataset/train/infected/non-covid',
                      'train_covid': './dataset/train/infected/covid',
                      }
    #
    # trainset = BinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)
    #
    val_groups = ['val']
    val_numbers = {'val_non-covid': 242,
                   'val_covid': 139,
                   }
    #
    valset_paths = {'val_non-covid': './dataset/test/infected/non-covid',
                    'val_covid': './dataset/test/infected/covid',
                    }
    #
    # valset = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)
    #
    # load dataset
    batch_size = 4
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    #
    epochs = 5
    model = Net(numberOfOutputLabels=2)
    #
    for epoch in range(1, epochs + 1):
        train(model, trainloader, epoch)
        validate(model, validationloader)
        torch.save(model.state_dict(), 'models/binaryModelCovid')

# DUPLICATE NORMAL
def train_binary_normal_clf(trainingEpochs, trainingBatchSize, savePath):

	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['train']
	dataset_numbers = {'train_normal': 0,
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

	valset_paths = {'val_normal': './dataset/val/normal',
	                'val_infected': './dataset/val/infected/covid',
	                }

	valset1 = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

	val_numbers = {'val_normal': 8,
	               'val_infected': 8,
	               }

	valset_paths = {'val_normal': './dataset/val/normal',
	                'val_infected': './dataset/val/infected/non-covid',
	                }

	valset2 = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

	# load dataset
	valsets = ConcatDataset([valset1, valset2])
	validationloader = DataLoader(valsets, batch_size=trainingBatchSize, shuffle=True)

	model = Net(numberOfOutputLabels=2)

	for epoch in range(1, trainingEpochs + 1):
		train(model, trainloader, epoch)
		validate(model, validationloader)
		torch.save(model.state_dict(), savePath)


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
	val_numbers = {'val_normal': 8,
				   'val_infected': 8,
				   'val_covid': 8,
				   }

	valset_paths = {'val_normal': './dataset/val/normal/',
					'val_infected': './dataset/val/infected/non-covid',
					'val_covid': './dataset/val/infected/covid',
					}

	valset = TrinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

	# load dataset
	trainloader = DataLoader(trainset, batch_size=trainingBatchSize, shuffle=True)
	validationloader = DataLoader(valset, batch_size=trainingBatchSize, shuffle=True)

	model = Net(3)

	for epoch in range(1, trainingEpochs + 1):
		train(model, trainloader, epoch)
		validate(model, validationloader)
		torch.save(model.state_dict(), savePath)



if __name__ == "__main__":

	trainingEpochs = 2
	trainingBatchSize = 4
	covidSavePath = 'models/binaryModelCovid'
	normalSavePath = 'models/binaryModelNormal'
	trinarySavePath = 'models/trinaryModel'

	train_binary_normal_clf(trainingEpochs, trainingBatchSize, normalSavePath)

	train_binary_covid_clf(trainingEpochs, trainingBatchSize, covidSavePath)

	train_trinary_clf(trainingEpochs, trainingBatchSize, trinarySavePath)

