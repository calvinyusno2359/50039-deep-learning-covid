import torch
import argparse

from model import Net
from dataset import BinaryClassDataset, TrinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset


# full of magic
# (bool) returnImmediate: asks if you need the image with predicted label matching intermediateLabel is to be returned
# (int) intermediateLabel: the particular label you are interested in retrieving the image for further processing
# (bool) isitBinary: binary has 2 pairs of labels, trinary only has 1
def test(model, testloader, returnIntermediate, intermediateLabel, isitBinary, device='cuda'):
	model.to(device)
	model.eval()
	accuracy = 0

	intermediate = []

	intermediateLabel = intermediateLabel.to(device)
	# first binary classifier
	if returnIntermediate and isitBinary:
		with torch.no_grad():
			for batch_idx, (images_data, target_labels, irrelevant) in enumerate(testloader):
				images_data, target_labels = images_data.to(device), target_labels.to(device)
				output = model(images_data)
				predicted_labels = torch.exp(output).max(dim=1)[1]
				print(predicted_labels, intermediateLabel)
				equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
				accuracy += equality.type(torch.FloatTensor).mean()

				if returnIntermediate and torch.equal(predicted_labels, intermediateLabel):
					intermediate.append([images_data, target_labels, irrelevant])

			print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))

	# second binary classifier
	elif not returnIntermediate and isitBinary:
		with torch.no_grad():
			for batch_idx, (images_data, irrelevant, target_labels) in enumerate(testloader):
				target_labels[0] = torch.narrow(target_labels[0], 0, 1, 1) # slicing the second bunch of labels
				images_data, target_labels = images_data.to(device), target_labels.to(device)
				output = model(images_data)
				predicted_labels = torch.exp(output).max(dim=1)[1]
				equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
				accuracy += equality.type(torch.FloatTensor).mean()

			print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))

	# trinary classifier
	else:
		with torch.no_grad():
			for batch_idx, (images_data, target_labels) in enumerate(testloader):
				images_data, target_labels = images_data.to(device), target_labels.to(device)
				output = model(images_data)
				predicted_labels = torch.exp(output).max(dim=1)[1]
				equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
				accuracy += equality.type(torch.FloatTensor).mean()

			print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))

	return intermediate


def get_args(argv=None):
	parser = argparse.ArgumentParser(description="test image classifier model")
	parser.add_argument("--output_var", type=int, default=2, help="number of output variables")
	parser.add_argument("--batch_size", type=int, default=1, help="set testing batch size")
	parser.add_argument("--gpu", action="store_const", const="cuda", default="cuda", help="use gpu")
	parser.add_argument("--load_from", type=str, help="specify path")
	return parser.parse_args(argv)

def run_double_binary():
	print('done')

if __name__ == "__main__":
	args = get_args()

	# set and load dataset spec
	img_size = (150, 150)

	if args.output_var == 2:
		img_size = (150, 150)

		print("normal binary classifier")

		normalCLFPath = 'models/binaryModelNormal_18_03_2021_15_29_50'

		class_dict = {0: 'normal', 1: 'infected'}
		groups = ['test']
		dataset_numbers = {'test_normal': 234,
						   'test_infected': 242,
						   }

		dataset_paths = {'test_normal': './dataset/test/normal/',
						 'test_infected': './dataset/test/infected/non-covid',
						 }

		# normal, infected
		# normal, non covid, covid
		testset1 = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

		test1 = []

		# appends an additional pair of labels
		# first pair: [normal, infected]
		# second pair: [normal, non-covid, covid] --> hardcoded tensor
		for i in range(len(testset1)):
			if torch.equal(testset1[i][1], torch.tensor([1., 0.])):
				test1.append((testset1[i][0], testset1[i][1], torch.tensor([1., 0., 0.])))
			elif torch.equal(testset1[i][1], torch.tensor([0., 1.])):
				test1.append((testset1[i][0], testset1[i][1], torch.tensor([0., 1., 0.])))

		dataset_numbers = {'test_normal': 0,
						   'test_infected': 138,
						   }

		dataset_paths = {'test_normal': './dataset/test/normal/',
						 'test_infected': './dataset/test/infected/covid',
						 }

		testset2 = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

		test2 = []

		# appends an additional pair of labels
		# first pair: [normal, infected]
		# second pair: [normal, non-covid, covid] --> hardcoded tensor
		for i in range(len(testset2)):
			if torch.equal(testset2[i][1], torch.tensor([0., 1.])):
				test2.append((testset2[i][0], testset2[i][1], torch.tensor([0., 0., 1.])))

		testsetNormal = ConcatDataset([test1, test2])
		testloaderNormal = DataLoader(testsetNormal, batch_size=args.batch_size, shuffle=True)

		model = Net(2)
		model.load_state_dict(torch.load(normalCLFPath))

		intermediate = test(model, testloaderNormal, True, torch.tensor([1.]), True)

		print("covid binary classifier (piped)")


		covidCLFPath = 'models/binaryModelCovid_18_03_2021_15_45_25'

		model = Net(numberOfOutputLabels=2)
		model.load_state_dict(torch.load(covidCLFPath))

		test(model, intermediate, False, torch.tensor([1.]), True)

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

