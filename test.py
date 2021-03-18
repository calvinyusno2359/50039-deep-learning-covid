import torch
import argparse

from model import Net
from dataset import BinaryClassDataset, TrinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset


# model: the model to be tested
# testloader: containing the test data
# desiredLabel: tensor of the label you want to pass to the second classifier
def test_first_binary(model, testloader, desiredLabel, device='cuda'):
	model.to(device)
	model.eval()
	accuracy = 0

	intermediate = []
	desiredLabel = desiredLabel.to(device)

	with torch.no_grad():
		for batch_idx, (images_data, target_labels, irrelevant) in enumerate(testloader):
			images_data, target_labels = images_data.to(device), target_labels.to(device)
			output = model(images_data)
			predicted_labels = torch.exp(output).max(dim=1)[1]
			equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
			accuracy += equality.type(torch.FloatTensor).mean()

			# if classified to be the one we are interested in
			if torch.equal(predicted_labels, desiredLabel):
				intermediate.append([images_data, target_labels, irrelevant])

	return intermediate


# model: the model to be tested
# testloader: containing the test data
# target_label issue
def test_second_binary(model, testloader, device='cuda'):
	model.to(device)
	model.eval()
	accuracy = 0

	with torch.no_grad():
		for batch_idx, (images_data, irrelevant, target_labels) in enumerate(testloader):
			target_labels[0] = torch.narrow(target_labels[0], 0, 1, 1)  # slicing the second bunch of labels
			images_data, target_labels = images_data.to(device), target_labels.to(device)
			output = model(images_data)
			predicted_labels = torch.exp(output).max(dim=1)[1]
			equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
			accuracy += equality.type(torch.FloatTensor).mean()

		print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))


# original test function
def test_original(model, testloader, device='cuda'):
	model.to(device)
	model.eval()
	accuracy = 0

	with torch.no_grad():
		for batch_idx, (images_data, target_labels) in enumerate(testloader):
			images_data, target_labels = images_data.to(device), target_labels.to(device)
			output = model(images_data)
			predicted_labels = torch.exp(output).max(dim=1)[1]
			equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
			accuracy += equality.type(torch.FloatTensor).mean()

		print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))


def get_args(argv=None):
	parser = argparse.ArgumentParser(description="test image classifier model")
	parser.add_argument("--output_var", type=int, default=2, help="number of output variables")
	parser.add_argument("--batch_size", type=int, default=1, help="set testing batch size")
	parser.add_argument("--gpu", action="store_const", const="cuda", default="cuda", help="use gpu")
	parser.add_argument("--load_from", type=str, help="specify path")
	return parser.parse_args(argv)


# magic inside
# adds a new set of label to each test sample
def __get_binary_piped_test_dataset(img_size, batch_size):
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
	testloaderNormal = DataLoader(testsetNormal, batch_size=batch_size, shuffle=True)

	return testloaderNormal


# independent normal dataset
def __get_binary_normal_test_dataset(img_size, batch_size):
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
	testloader = DataLoader(testsets, batch_size=batch_size, shuffle=True)

	return testloader


# independent covid dataset
def __get_binary_covid_test_dataset(img_size, batch_size):
	class_dict = {0: 'non-covid', 1: 'covid'}
	dataset_numbers = {'test_non-covid': 242,
					   'test_covid': 138,
					   }

	dataset_paths = {'test_non-covid': './dataset/test/infected/non-covid',
					 'test_covid': './dataset/test/infected/covid',
					 }

	testset = BinaryClassDataset('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)

	testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
	return testloader


# trinary dataset
def __get_trinary_test_dataset(img_size, batch_size):
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

	testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

	return testloader


if __name__ == "__main__":
	args = get_args()

	# set and load dataset spec
	img_size = (150, 150)

	# model parameters
	normalCLFPath = 'models/binaryModelNormalBest'
	covidCLFPath = 'models/binaryModelCovidBest'
	trinaryCLFPath = 'models/trinaryModel'

	# if you want independent or piped binary classifier
	independent = False

	# doing binary classifier
	if args.output_var == 2:

		if not independent:

			print("Starting: Normal piped binary classifier")

			# get test loader
			testloaderNormal = __get_binary_piped_test_dataset(img_size, arg.batch_size)

			# define model
			model = Net(2)

			# fetch model saved state
			model.load_state_dict(torch.load(normalCLFPath))

			# test and get the intermediate dataset for second classifier
			intermediateTestLoader = test_first_binary(model, testloaderNormal, torch.tensor[1.])

			print("Starting: Covid piped binary classifier")

			# fetch model saved state
			model.load_state_dict(torch.load(covidCLFPath))

			# test and print
			test_second_binary(model, intermediateTestLoader)

		else:
			print("Starting: Normal independent binary classifier")

			normalIndependentTestloader = __get_binary_normal_test_dataset(img_size, arg.batch_size)

			model = Net(2)
			model.load_state_dict(torch.load(normalCLFPath))
			test_original(model, normalIndependentTestloader)

			print("Starting: Covid independent binary classifier")

			covidIndependentTestloader = __get_binary_covid_test_dataset(img_size, arg.batch_size)

			model.load_state_dict(torch.load(covidCLFPath))
			test_original(model, covidIndependentTestloader)

	elif args.output_var == 3:

		print("Starting: Trinary classifier")

		trinaryTestloader = __get_trinary_test_dataset(img_size)

		model = Net(3)
		model.load_state_dict(torch.load(trinaryCLFPath))
		test_original(model, trinaryTestloader)

	else:
		print("only 2 or 3")

