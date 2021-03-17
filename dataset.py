import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Image_Dataset(Dataset):
	def __init__(self, img_size, class_dict, groups, dataset_numbers, dataset_paths):
		self.img_size = img_size
		self.classes = class_dict
		self.groups = groups
		self.dataset_numbers = dataset_numbers
		self.dataset_paths = dataset_paths

	def describe(self):
		msg = f"""
		It contains a total of {sum(self.dataset_numbers.values())} images of size {self.img_size}.
		Images have been split in {len(self.groups)} groups: {self.groups} sets.
		The images are stored in the following locations, each containing the following images:
		"""
		print(msg)

		for key, val in self.dataset_paths.items():
			print(f" - {key}, in folder {val}: {self.dataset_numbers[key]} images.")

	def open_img(self, group_val, class_val, index_val):
		err_msg = f"Error - group_val variable should be set to {self.groups}."
		assert group_val in self.groups, err_msg

		err_msg = f"Error - class_val variable should be set to {self.classes.values()}."
		assert class_val in self.classes.values(), err_msg

		max_val = self.dataset_numbers[f'{group_val}_{class_val}']
		err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
		err_msg += f"\n(In {group_val}/{class_val}, you have {max_val} images.)"
		assert isinstance(index_val, int), err_msg
		assert index_val >= 0 and index_val <= max_val, err_msg

		dataset_path = self.dataset_paths[f'{group_val}_{class_val}']
		path_to_file = f'{dataset_path}/{index_val}.jpg'

		with open(path_to_file, 'rb') as f:
		    im = np.asarray(Image.open(f)) / 255
		f.close()
		return im

	def show_img(self, group_val, class_val, index_val):
		im = self.open_img(group_val, class_val, index_val)
		plt.imshow(im)


class Image_Dataset_Part(Image_Dataset):

	def __init__(self, title, img_size, class_dict, groups, dataset_numbers, dataset_paths):
	    super().__init__(img_size, class_dict, groups, dataset_numbers, dataset_paths)
	    self.title = title

	def __len__(self):
	    return sum(self.dataset_numbers.values())

	def __getitem__(self, index):
		first_val = int(list(self.dataset_numbers.values())[0])
		# pre = index
		second_val = int(list(self.dataset_numbers.values())[1])
		if index < first_val:
			class_val = 'normal'
			label = torch.Tensor([1, 0, 0])
		elif first_val <= index < (second_val + first_val):
			class_val = 'infected'
			index = index - first_val
			label = torch.Tensor([0, 1, 0])
		else:
			class_val = 'covid'
			index = index - (first_val + second_val)
			label = torch.Tensor([0, 0, 1])
		# print(pre, index)
		im = self.open_img(self.groups[0], class_val, index)
		im = transforms.functional.to_tensor(np.array(im)).float()
		return im, label

if __name__ == "__main__":
	dataset_path = './dataset_demo'

	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['train', 'test', 'val']
	dataset_numbers = {'train_normal': 36,
	                   'train_infected': 34,
	                   'val_normal': 4,
	                   'val_infected': 4,
	                   'test_normal': 14,
	                   'test_infected': 13
	                   }

	dataset_paths = {'train_normal': './dataset_demo/train/normal/',
	                 'train_infected': './dataset_demo/train/infected/',
	                 'val_normal': './dataset_demo/val/normal/',
	                 'val_infected': './dataset_demo/val/infected/',
	                 'test_normal': './dataset_demo/test/normal/',
	                 'test_infected': './dataset_demo/test/infected/'
	                 }

	dataset = Image_Dataset(img_size, class_dict, groups, dataset_numbers, dataset_paths)
	dataset.describe()
	im = dataset.open_img('train', 'normal', 1)
	print(im.shape)
	print(im)
	dataset.show_img('train', 'normal', 1)
	plt.show()
