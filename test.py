import torch
import argparse

from model import Net
from dataset import Image_Dataset_Part
from torch.utils.data import DataLoader

def test(model, testloader, device='cuda'):
	model.to(device)
	model.eval()
	accuracy = 0

	# Try model on one mini-batch
	for batch_idx, (images_data, target_labels) in enumerate(testloader):
		images_data, target_labels = images_data.to(device), target_labels.to(device)
		output = model(images_data)
		predicted_labels = torch.exp(output).max(dim=1)[1]
		equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
		accuracy += equality.type(torch.FloatTensor).mean()

	print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))

def get_args(argv=None):
	parser = argparse.ArgumentParser(description="test image classifier model")
	parser.add_argument("--batch_size", type=int, default=4, help="set batch size")
	parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
	return parser.parse_args(argv)

if __name__ == "__main__":
	args = get_args()
	# set and load dataset spec
	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['test']
	dataset_numbers = { 'test_normal': 14,
											'test_infected': 13,
										}

	dataset_paths = { 'test_normal': './dataset_demo/test/normal/',
										'test_infected': './dataset_demo/test/infected/',
									}

	testset = Image_Dataset_Part('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

	# load dataset
	testloader = DataLoader(testset, batch_size = args.batch_size, shuffle = True)

	# load model
	model = Net()

	test(model, testloader)
