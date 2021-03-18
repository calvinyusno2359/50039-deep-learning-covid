import torch
import argparse

from model import Net
from dataset import BinaryClassDataset, TrinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset


# full of magic
# (bool) returnImmediate: asks if you need the image with predicted label matching intermediateLabel is to be returned
# (int) intermediateLabel: the particular label you are interested in retrieving the image for further processing
# (bool) isitBinary: binary has 2 pairs of labels, trinary only has 1
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
	parser.add_argument("--output_var", type=int, default=2, help="number of output variables")
	parser.add_argument("--batch_size", type=int, default=1, help="set testing batch size")
	parser.add_argument("--gpu", action="store_const", const="cuda", default="cuda", help="use gpu")
	parser.add_argument("--load_from", type=str, help="specify path")
	return parser.parse_args(argv)

def run_double_binary(normal_path, covid_path, device='cuda'):
	# normal
	img_size = (150, 150)
	groups = ['test']

	class_dict = {0: 'normal', 1: 'infected'}
	dataset_numbers = {'test_normal': 234,
					   'test_infected': 242,
					   }

	dataset_paths = {'test_normal': './dataset/test/normal/',
					 'test_infected': './dataset/test/infected/non-covid',
					 }

	testset1 = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

	dataset_numbers = {'test_normal': 0,
					   'test_infected': 138,
					   }

	dataset_paths = {'test_normal': './dataset/test/normal/',
					 'test_infected': './dataset/test/infected/covid',
					 }

	testset2 = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

	# load dataset
	testsets = ConcatDataset([testset1, testset2])
	testloader = DataLoader(testsets, batch_size=4, shuffle=True)

	model = Net(2)
	model.load_state_dict(torch.load(normal_path))
	test(model, testloader, device)

	# covid
	class_dict = {0: 'non-covid', 1: 'covid'}
	dataset_numbers = {'test_non-covid': 242,
					   'test_covid': 138,
					   }

	dataset_paths = {'test_non-covid': './dataset/test/infected/non-covid',
					 'test_covid': './dataset/test/infected/covid',
					 }

	testset = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

	testloader = DataLoader(testset, batch_size=4, shuffle=True)

	model = Net(2)
	model.load_state_dict(torch.load(covid_path))
	test(model, testloader, device)

	print('done')

if __name__ == "__main__":
	args = get_args()

	# model parameters
	covidCLFPath = 'models/binaryModelCovid_18_03_2021_15_45_25'
	normalCLFPath = 'models/binaryModelNormal_18_03_2021_15_29_50'

	# doing binary classifier
	if args.output_var == 2:
		print("Starting: Double Binary Classifier Test")
		run_double_binary(normalCLFPath, covidCLFPath)

	elif args.output_var == 3:
		print("trinary classifier")
		trinaryCLFPath = 'models/trinaryModel'

		class_dict = {0: 'normal', 1: 'infected', 2: 'covid'}

		test_groups = ['test']
		test_numbers = {'test_normal': 234,
						'test_infected': 242,
						'test_covid': 138,
						}

		testset_paths = {'test_normal': './dataset/test/normal/',
						 'test_infected': './dataset/test/infected/non-covid',
						 'test_covid': './dataset/test/infected/covid',
						 }

		testset = TrinaryClassDataset('test', img_size, class_dict, test_groups, test_numbers, testset_paths)

		testloader = DataLoader(testset, batch_size=1, shuffle=True)
		model = Net(3)
		model.load_state_dict(torch.load(trinaryCLFPath))
		test(model, testloader, False, torch.tensor([1.]), False)

	else:
		print("only 2 or 3")

