import torch
import argparse

from model import Net
from dataset import BinaryClassDataset, TrinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset


def test(model, testloader, device='cuda'):
	model.to(device)
	model.eval()
	accuracy = 0

	# Try model on one mini-batch
	with torch.no_grad():
		for batch_idx, (images_data, target_labels) in enumerate(testloader):
			images_data, target_labels = images_data.to(device), target_labels.to(device)
			output = model(images_data)
			predicted_labels = torch.exp(output).max(dim=1)[1]
			equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
			print(equality)
			accuracy += equality.type(torch.FloatTensor).mean()

		print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))


def get_args(argv=None):
	parser = argparse.ArgumentParser(description="test image classifier model")
	parser.add_argument("--output_var", type=int, default=2, help="number of output variables")
	parser.add_argument("--batch_size", type=int, default=4, help="set testing batch size")
	parser.add_argument("--gpu", action="store_const", const="cuda", default="cuda", help="use gpu")
	parser.add_argument("--load_from", type=str, help="specify path")
	return parser.parse_args(argv)


if __name__ == "__main__":
	# args = get_args()

	# set and load dataset spec
	img_size = (150, 150)

	if args.output_var == 2:
		normalCLFPath = 'models/binaryModelNormal'

		print("normal classifier")
		class_dict = {0: 'normal', 1: 'infected'}
		groups = ['test']
		dataset_numbers = {'test_normal': 234,
						   'test_infected': 242,
						   }

		dataset_paths = {'test_normal': './dataset/test/normal/',
						 'test_infected': './dataset/test/infected/non-covid',
						 }

		testset1 = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

		dataset_numbers = {'test_normal': 234,
						   'test_infected': 138,
						   }

		dataset_paths = {'test_normal': './dataset/test/normal/',
						 'test_infected': './dataset/test/infected/covid',
						 }

		testset2 = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

		# load dataset
		testsets = ConcatDataset([testset1, testset2])
		#testloader = DataLoader(testsets, batch_size=args.batch_size, shuffle=True)
		testloader = DataLoader(testsets, batch_size=4, shuffle=True)

		# load model
		model = Net(2)
		# if args.load_from is not None:
		# 	model.load_state_dict(torch.load(args.load_from))
		model.load_state_dict(torch.load(normalCLFPath))

		test(model, testloader)


		print("covid classifier")
		covidCLFPath = 'models/binaryModelCovid'
		class_dict = {0: 'non-covid', 1: 'covid'}
		test_groups = ['test']
		test_numbers = {'test_non-covid': 242,
					   'test_covid': 138,
					   }

		testset_paths = {'test_non-covid': './dataset/test/infected/non-covid',
						'test_covid': './dataset/test/infected/covid',
						}

		testset = BinaryClassDataset('test', img_size, class_dict, test_groups, test_numbers, testset_paths)

		# load dataset
		testloader = DataLoader(testset, batch_size=4, shuffle=True)

		model = Net(numberOfOutputLabels=2)

		model.load_state_dict(torch.load(covidCLFPath))

		test(model, testloader)

		# ?? how to use the output and why are there 4 labels??


	elif args.output_var == 3:
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

		# load dataset
		testloader = DataLoader(testset, batch_size=4, shuffle=True)

		model = Net(3)

		model.load_state_dict(torch.load(trinaryCLFPath))

		test(model, testloader)

	else:
		print("only 2 or 3")

		# normal
		# binary
		# classifier
		# Testing
		# Accuracy: 0.759
		# covid
		# binary
		# classifier(independent)
		# Testing
		# Accuracy: 0.426
		# trinary
		# classifier
		# Testing
		# Accuracy: 0.718

