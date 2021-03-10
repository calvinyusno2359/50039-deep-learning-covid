import time
import torch

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

def train(model, trainloader, validationloader, epochs, device='cuda'):
	for param in model.parameters():
	    param.requires_grad = False

	# Define loss function and optimizer
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.001)

	model.to(device)
	start = time.time()

	epochs = epochs
	steps = 0
	running_loss = 0
	print_every = 1
	training_losses = []
	for e in range(epochs):
		model.train()
		for images, labels in trainloader:
			images, labels = images.to(device), labels.to(device)

			steps += 1

			optimizer.zero_grad()

			output = model.forward(images)
			loss = loss_function(output, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if steps % print_every == 0:
				# Eval mode for predictions
				model.eval()

				# Turn off gradients for validation
				with torch.no_grad():
				    test_loss, accuracy = validate(model, validationloader, criterion, device)

				print("Epoch: {}/{} - ".format(e+1, epochs),
				      "Training Loss: {:.3f} - ".format(running_loss/print_every),
				      "Validation Loss: {:.3f} - ".format(test_loss/len(validationloader)),
				      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

				running_loss = 0

				model.train()

	print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
	print(f"Run time: {(time.time() - start)/60:.3f} min")
	return model, training_losses, validation_losses

if __name__ == "__main__":
	# set and load dataset spec
	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	train_groups = ['train']
	train_numbers = { 'train_normal': 36,
											'train_infected': 34,
										}

	trainset_paths = { 'train_normal': './dataset_demo/train/normal/',
										'train_infected': './dataset_demo/train/infected/',
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

	train(model, trainloader, validationloader, epochs)
