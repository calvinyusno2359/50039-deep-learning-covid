import time
import torch
import numpy as np
import torch.nn.functional as F

from model import Net

from torch import nn
from torch import optim
from torchvision import models
from dataset import Image_Dataset_Part
from torch.utils.data import DataLoader


def validate(model, validloader, criterion, device='cuda'):
	model.to(device)
	model.eval()
	test_loss = 0
	accuracy = 0

	for batch_idx, (images_data, target_labels) in enumerate(validloader):
		images_data, target_labels = images_data.to(device), target_labels.to(device)
		output = model(images_data)

		test_loss += criterion(output, target_labels).item()

		predicted_labels = torch.exp(output).max(dim=1)[1]

		equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
		accuracy += equality.type(torch.FloatTensor).mean()

	return test_loss, accuracy

def train(model, trainloader, epochs, device='cuda'):
	print('training now')
	model.to(device)
	model.train()
	for batch_idx, (data, target) in enumerate(trainloader):
		target = np.argmax(target, axis=1)
		print(target)
		data, target = data.to(device), target.to(device)
		optimizer = optim.Adadelta(model.parameters(), lr=0.001)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 2 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epochs, batch_idx * len(data), len(trainloader.dataset),
				100. * batch_idx / len(trainloader), loss.item()))

if __name__ == "__main__":
	# set and load dataset spec
	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected', 2:'covid'}
	train_groups = ['train']
	train_numbers = { 'train_normal': 1341,
											'train_infected': 2530,
											'train_covid': 1345
										}

	trainset_paths = { 'train_normal': './dataset/train/normal/',
										'train_infected': './dataset/train/infected/',
										'train_covid': './dataset/train/covid/'
									}

	trainset = Image_Dataset_Part('train', img_size, class_dict, train_groups, train_numbers, trainset_paths)

	val_groups = ['val']
	val_numbers = { 'val_normal': 4,
											'val_infected': 4,
										}

	valset_paths = { 'val_normal': './dataset_demo/val/normal/',
										'val_infected': './dataset_demo/val/infected/',
									}

	valset = Image_Dataset_Part('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

	# load dataset
	batch_size = 4
	trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
	validationloader = DataLoader(valset, batch_size = batch_size, shuffle = True)

	epochs = 2
	model = Net()

	train(model, trainloader, epochs)
