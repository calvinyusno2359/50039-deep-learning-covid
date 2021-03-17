import time
import torch
import numpy as np
import torch.nn.functional as F

from model import Net, DenseNet

from torch import nn
from torch import optim
from torchvision import models
from dataset import TrinaryClassDataset, BinaryClassDataset
from torch.utils.data import DataLoader


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


# if __name__ == "__main__":
# 	# set and load dataset spec
# 	img_size = (150, 150)
# 	class_dict = {0: 'normal', 1: 'infected', 2: 'covid'}
# 	train_groups = ['train']
# 	train_numbers = {'train_normal': 1341,
# 	                 'train_infected': 2530,
# 	                 'train_covid': 1345
# 	                 }
#
# 	trainset_paths = {'train_normal': './dataset/train/normal',
# 	                  'train_infected': './dataset/train/infected/non-covid',
# 	                  'train_covid': './dataset/train/infected/covid'
# 	                  }
#
# 	trainset = TrinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)
#
# 	val_groups = ['val']
# 	val_numbers = {'val_normal': 234,
# 	               'val_infected': 242,
# 	               'val_covid': 139,
# 	               }
#
# 	valset_paths = {'val_normal': './dataset/test/normal/',
# 	                'val_infected': './dataset/test/infected/non-covid',
# 	                'val_covid': './dataset/test/infected/covid',
# 	                }
#
# 	valset = TrinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)
#
# 	# load dataset
# 	batch_size = 4
# 	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# 	validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
#
# 	epochs = 5
# 	# model = DenseNet(3) #39%
# 	model = Net(3)
#
# 	for epoch in range(1, epochs + 1):
# 	    train(model, trainloader, epoch, device="cpu")
# 	    validate(model, validationloader, device="cpu")


if __name__ == "__main__":
	# set and load dataset spec
	img_size = (150, 150)
	class_dict = {0: 'non-covid', 1: 'covid'}
	train_groups = ['train']
	train_numbers = {'train_non_covid': 2530,
	                 'train_covid': 1345
	                 }

	trainset_paths = {'train_non_covid': './dataset/train/infected/non-covid',
	                  'train_covid': './dataset/train/infected/covid'
	                  }

	trainset = BinaryClassDataset('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)

	val_groups = ['val']
	val_numbers = {'val_non_covid': 242,
	               'val_covid': 139,
	               }

	valset_paths = {'val_non_covid': './dataset/test/infected/non-covid',
	                'val_covid': './dataset/test/infected/covid',
	                }

	valset = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

	# load dataset
	batch_size = 4
	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
	validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

	epochs = 1
	# model = DenseNet(3) #39%
	model = Net(3)

	for epoch in range(1, epochs + 1):
	    train(model, trainloader, epoch)
	    validate(model, validationloader)
