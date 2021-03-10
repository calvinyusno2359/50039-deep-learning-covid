import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Image_Dataset_Part
from torch.utils.data import DataLoader

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

if __name__ == "__main__":
	# Create model
	model = Net()

	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['train']
	dataset_numbers = { 'train_normal': 36,
											'train_infected': 34,
										}

	dataset_paths = { 'train_normal': './dataset_demo/train/normal/',
										'train_infected': './dataset_demo/train/infected/',
									}

	bs_val = 4
	ld_train = Image_Dataset_Part('train', img_size, class_dict, groups, dataset_numbers, dataset_paths)
	train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)

	# Try model on one mini-batch
	for batch_idx, (images_data, target_labels) in enumerate(train_loader):
	    predicted_labels = model(images_data)
	    print(predicted_labels)
	    print(target_labels)
	    # Forced stop
	    break
	    #assert False, "Forced stop after one iteration of the mini-batch for loop"
