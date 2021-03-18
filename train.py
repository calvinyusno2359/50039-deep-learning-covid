import torch
import numpy as np
import torch.nn.functional as F

from datetime import datetime
from model import Net, DenseNet

from torch import nn
from torch import optim
from torchvision import models, transforms
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
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			test_loss /= len(validloader.dataset)

	print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(validloader.dataset),
		100. * correct / len(validloader.dataset)))

def transform(img_tensor):
	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(45),
	])

	return transform(img_tensor)

def train(model, trainloader, epoch, device='cuda'):
	print(f'Train Epoch: {epoch}')
	model.to(device)
	model.train()
	for batch_idx, (data, target) in enumerate(trainloader):
		target = np.argmax(target, axis=1)
		data, target = data.to(device), target.to(device)
		data = transform(data)
		optimizer = optim.Adadelta(model.parameters(), lr=0.001)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		# TODO Change Batch size printing
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(trainloader.dataset),
						100. * batch_idx / len(trainloader), loss.item()))

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

def train_binary_normal_clf(trainingEpochs, trainingBatchSize, savePath):
	# normal vs infected clf
	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['train']

	dataset_numbers = {'train_normal': 0, # so it does not double count
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
	now = datetime.now()
	timestamp = now.strftime("_%d_%m_%Y_%H_%M_%S")

	trainingEpochs = 7
	trainingBatchSize = 4
	# covidSavePath = f'models/binaryModelCovid{timestamp}'
	normalSavePath = f'models/binaryModelNormal{timestamp}'
	# trinarySavePath = f'models/trinaryModel{timestamp}'

	train_binary_normal_clf(trainingEpochs, trainingBatchSize, normalSavePath)

	# train_binary_covid_clf(trainingEpochs, trainingBatchSize, covidSavePath)

	# train_trinary_clf(trainingEpochs, trainingBatchSize, trinarySavePath)

