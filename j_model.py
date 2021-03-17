import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import Image_Dataset_Part
from torch.utils.data import DataLoader
from utils import save_model, load_model


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 4, 3, 1)
		self.fc1 = nn.Linear(87616, 2)

	def forward(self, x):
		x = self.conv1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		output = F.log_softmax(x, dim = 1)
		return output


def train_model(model, train_loader, optimiser, epochs, path, saving):
	print("training now")
	model.train()
	for i in range(epochs):
		for batch_idx, (data, target) in enumerate(train_loader):
			output = model(data)
			print("loss: {:.3f}")
			print("predicted:", output)
			print("target:", target)
	if saving:
		save_model(model, path)

def test_model(model, test_loader):
	model.eval()
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = model(data)
			pred = output.argmax(dim=1, keepdim=True)


if __name__ == "__main__":
	# Create model
	model = Net()

	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['train']
	dataset_numbers = { 'train_normal': 36,
						'train_infected': 34,}

	dataset_paths = { 'train_normal': './dataset_demo/train/normal/',
										'train_infected': './dataset_demo/train/infected/',
									}

	bs_val = 4
	ld_train = Image_Dataset_Part('train', img_size, class_dict, groups, dataset_numbers, dataset_paths)
	train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)


	# training
	optimiser = optim.Adam(model.parameters())

	# check training and saving
	train_model(model, train_loader, optimiser, 1, "./checkpoints/hello", True)

	# check loading
	model = load_model(model, "./checkpoints/hello")
	test_model(model, train_loader)
