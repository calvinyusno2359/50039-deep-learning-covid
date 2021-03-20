import torch
import argparse

from model import ResNet
from dataset import BinaryClassDataset, TrinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset
from torchvision import transforms

def transform(img_tensor):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4824], std=[0.2363]),
    ])

    return transform(img_tensor)

# model: the model to be tested
# testloader: containing the test data
# desiredLabel: tensor of the label you want to pass to the second classifier. this will also be the "positive"
def test_first_binary(model, testloader, desiredLabel, device='cpu'):
    model.to(device)
    model.eval()
    # accuracy = 0

    intermediate = []
    desiredLabel = desiredLabel.to(device)

    TP, FP, FN, TN = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (images_data, target_labels, irrelevant) in enumerate(testloader):
            images_data, target_labels = images_data.to(device), target_labels.to(device)
            images_data = transform(images_data)
            output = model(images_data)
            predicted_labels = torch.max(output, 1)[1] # get prediction

            for j in range(images_data.size()[0]):
                # if this is the sample with the label that we are interested in processing further
                if torch.equal(predicted_labels[j], desiredLabel[0]):

                    # append to intermediate dataloader
                    intermediate.append([images_data, target_labels, irrelevant])

                    # true positive
                    if torch.equal(target_labels[j].data.max(dim=0)[1], desiredLabel[0]):
                        TP += 1

                    # false positive
                    else:
                        FP += 1

                # negative
                else:
                    # false negative
                    if torch.equal(target_labels[j].data.max(dim=0)[1], desiredLabel[0]):
                        FN += 1

                    # true negative
                    else:
                        TN += 1

    accuracy = (TP + TN) / len(testloader)
    sensitivity = TP / (TP + FN) # WHAT WE WANT
    specificity = TN / (TN + FP) # WHAT WE WANT
    if (TP + FP) == 0:
        ppv = 0
    else:
        ppv = TP / (TP + FP)  # how many positives were correct
    if (TN + FN) == 0:
        npv = 0
    else:
        npv = TN / (TN + FN) # how many negatives were correct
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) #balance view of model
    print(f"Total={len(testloader)}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print('Testing Accuracy: {:.3f}'.format(accuracy))
    print('Testing Sensitivity: {:.3f}'.format(sensitivity))
    print('Testing Specificity: {:.3f}'.format(specificity))
    print('Testing PPV: {:.3f}'.format(ppv))
    print('Testing NPV: {:.3f}'.format(npv))
    print('Testing F1 Score: {:.3f}'.format(f1))

    return intermediate


# model: the model to be tested
# testloader: containing the test data
# desiredLabel: tensor of the label you want to track as positive. in this case it will be for true covid cases
# target_label issue
def test_second_binary(model, testloader, desiredLabel, device='cpu'):
    model.to(device)
    model.eval()
    accuracy = 0

    desiredLabel = desiredLabel.to(device)

    TP, FP, FN, TN = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (images_data, irrelevant, target_labels) in enumerate(testloader):
            target_labels = torch.narrow(target_labels[0], 0, 1, 2)  # slicing the second bunch of labels
            images_data, target_labels = images_data.to(device), target_labels.to(device)
            images_data = transform(images_data)
            output = model(images_data)
            predicted_labels = torch.max(output, 1)[1]

            for j in range(images_data.size()[0]):

                if torch.equal(predicted_labels[j], desiredLabel[0]):

                    # true positive
                    if torch.equal(target_labels.data.max(dim=0)[1], desiredLabel[0]):
                        TP += 1

                    # false positive
                    else:
                        FP += 1

                # negative
                else:
                    # false negative
                    if torch.equal(target_labels.data.max(dim=0)[1], desiredLabel[0]):
                        FN += 1

                    # true negative
                    else:
                        TN += 1

    accuracy = (TP + TN) / len(testloader)
    sensitivity = TP / (TP + FN)  # WHAT WE WANT
    specificity = TN / (TN + FP)  # WHAT WE WANT
    if (TP + FP) == 0:
        ppv = 0
    else:
        ppv = TP / (TP + FP)  # how many positives were correct
    if (TN + FN) == 0:
        npv = 0
    else:
        npv = TN / (TN + FN) # how many negatives were correct
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity)  # balance view of model
    print(f"Total={len(testloader)}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    # print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))
    print('Testing Accuracy: {:.3f}'.format(accuracy))
    print('Testing Sensitivity: {:.3f}'.format(sensitivity))
    print('Testing Specificity: {:.3f}'.format(specificity))
    print('Testing PPV: {:.3f}'.format(ppv))
    print('Testing NPV: {:.3f}'.format(npv))
    print('Testing F1 Score: {:.3f}'.format(f1))


# original test function
def test_original(model, testloader, device='cuda'):
    model.to(device)
    model.eval()
    # accuracy = 0
    TP, FP, FN, TN = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (images_data, target_labels) in enumerate(testloader):
            images_data, target_labels = images_data.to(device), target_labels.to(device)
            images_data = transform(images_data)
            output = model(images_data)
            # predicted_labels = torch.exp(output).max(dim=1)[1]
            predicted_labels = torch.max(output, 1)[1]  # shortform
            # equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
            # accuracy += equality.type(torch.FloatTensor).mean()
            for j in range(images_data.size()[0]):

                # True Positive (infected)
                if predicted_labels[j] == 1 and torch.argmax(target_labels[j]) == 1:
                    TP += 1

                # False Positive (should be normal)
                elif predicted_labels[j] == 1 and torch.argmax(target_labels[j]) == 0:
                    FP += 1

                # False Negative (should be infected)
                elif predicted_labels[j] == 0 and torch.argmax(target_labels[j]) == 1:
                    FN += 1

                # True Negative (normal)
                elif predicted_labels[j] == 0 and torch.argmax(target_labels[j]) == 0:
                    TN += 1

    accuracy = (TP + TN) / len(testloader)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity)
    print(f"Total={len(testloader)}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    # print('Testing Accuracy: {:.3f}'.format(accuracy / len(testloader)))
    print('Testing Accuracy: {:.3f}'.format(accuracy))
    print('Testing Sensitivity: {:.3f}'.format(sensitivity))
    print('Testing Specificity: {:.3f}'.format(specificity))
    print('Testing PPV: {:.3f}'.format(ppv))
    print('Testing NPV: {:.3f}'.format(npv))
    print('Testing F1 Score: {:.3f}'.format(f1))


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
    groups = ['test']
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
    output_var = 2

    # set and load dataset spec
    img_size = (150, 150)

    # model parameters
    covidCLFPath = 'models/binaryModelCovid2103_0101_5'
    normalCLFPath = 'models/binaryModelNormal2103_0101_11'
    trinaryCLFPath = 'models/trinaryModel'

    # if you want independent or piped binary classifier
    independent = True

    # doing binary classifier
    # if args.output_var == 2:
    if output_var == 2:

        if not independent:

            print("Starting: Normal piped binary classifier")

            # get test loader
            # testloaderNormal = __get_binary_piped_test_dataset(img_size, args.batch_size)
            testloaderNormal = __get_binary_piped_test_dataset(img_size, 1)

            # define model
            model = ResNet(2)

            # fetch model saved state
            # model.load_state_dict(torch.load(normalCLFPath))
            model.load_state_dict(torch.load(normalCLFPath, map_location=torch.device('cpu')))

            # test and get the intermediate dataset for second classifier
            # label (normal, infected)
            intermediateTestLoader = test_first_binary(model, testloaderNormal, torch.tensor([1.]).type(torch.int64))

            # looking into the contents of intermediate (the samples that got piped through
            yes = torch.tensor([0., 1.])
            no = torch.tensor([1., 0.])
            yesCounter = 0
            noCounter = 0
            print("intermediate length", len(intermediateTestLoader))
            for i in range(len(intermediateTestLoader)):
                if torch.equal(intermediateTestLoader[i][1][0], yes):
                    yesCounter += 1
                elif torch.equal(intermediateTestLoader[i][1][0], no):
                    noCounter += 1
            print("wrong: normal in intermediate", noCounter)
            print("correct: infected in intermediate", yesCounter)

            print("Starting: Covid piped binary classifier")

            # fetch model saved state
            # model.load_state_dict(torch.load(covidCLFPath))
            model.load_state_dict(torch.load(covidCLFPath, map_location=torch.device('cpu')))

            # test and print
            # label (noncovid, covid)
            test_second_binary(model, intermediateTestLoader, torch.tensor([1.]).type(torch.int64))

        else:
            print("Starting: Normal independent binary classifier")

            normalIndependentTestloader = __get_binary_normal_test_dataset(img_size, 1)

            model = ResNet(2)
            model.load_state_dict(torch.load(normalCLFPath))
            test_original(model, normalIndependentTestloader)

            print("Starting: Covid independent binary classifier")

            covidIndependentTestloader = __get_binary_covid_test_dataset(img_size, 1)

            model = ResNet(2)
            model.load_state_dict(torch.load(covidCLFPath))
            test_original(model, covidIndependentTestloader)

    # elif args.output_var == 3:
    elif output_var == 3:

        print("Starting: Trinary classifier")

        trinaryTestloader = __get_trinary_test_dataset(img_size)

        model = Net(3)
        model.load_state_dict(torch.load(trinaryCLFPath))
        test_original(model, trinaryTestloader)

    else:
        print("only 2 or 3")

