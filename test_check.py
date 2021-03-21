import torch
import argparse

from model import ResNet
from torchvision import transforms
from dataset import BinaryClassDataset, TrinaryClassDataset
from torch.utils.data import DataLoader, ConcatDataset, ChainDataset
import matplotlib.pyplot as plt


# normalisation applied on to the test data
def transform(img_tensor):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4824], std=[0.2363])
    ])

    return transform(img_tensor)


# returns 0 if division by 0
def safe_division(a, b):
    return a/b if b else 0

def transform(img_tensor):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.4824], std=[0.2363]),
    ])

    return transform(img_tensor)

# model: the model to be tested
# testloader: containing the test data
# desiredLabel: tensor of the label you want to pass to the second classifier. this will also be the "positive"
# displayPrint: bool to tell if the print statements are to be shown or not
# validation: bool to return values into the validation set for display
def test_first_binary(model, testloader, desiredLabel, displayPrint, validation, device='cpu'):
    model.to(device)
    model.eval()

    intermediate = []
    valid = []
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
                    if validation:
                        valid.append((images_data[j], target_labels[j].data.max(dim=0)[1].type(torch.int64), predicted_labels[j]))

                    # false negative
                    if torch.equal(target_labels[j].data.max(dim=0)[1], desiredLabel[0]):
                        FN += 1

                    # true negative
                    else:
                        TN += 1

    accuracy = (TP + TN) / len(testloader)
    sensitivity = safe_division(TP, (TP+FN))  # WHAT WE WANT
    specificity = safe_division(TN, (TN+FP))  # WHAT WE WANT
    ppv = safe_division(TP, (TP+FP))
    npv = safe_division(TN, (TN+FN))

    f1 = safe_division((2 * (ppv * sensitivity)), (ppv + sensitivity))  # balance view of model

    if displayPrint:
        print(f"Total={len(testloader)}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
        print('Testing Accuracy: {:.3f}'.format(accuracy))
        print('Testing Sensitivity: {:.3f}'.format(sensitivity))
        print('Testing Specificity: {:.3f}'.format(specificity))
        print('Testing PPV: {:.3f}'.format(ppv))
        print('Testing NPV: {:.3f}'.format(npv))
        print('Testing F1 Score: {:.3f}'.format(f1))
        print('Confusion Matrix')
        print('          Predicted N | Predicted P')
        print('         ---------------------------')
        print('Ground N |   TN = {}  |   FP = {}   |').format(TN, FP)
        print('Ground P |   FN = {}  |   TP = {}   |').format(FN, TP)
        print('\n')

    return intermediate, valid


# model: the model to be tested
# testloader: containing the test data
# desiredLabel: tensor of the label you want to track as positive. in this case it will be for true covid cases
# displayPrint: bool to tell if the print statements are to be shown or not
# validation: bool to return values into the validation set for display
def test_second_binary(model, testloader, desiredLabel, displayPrint, validation, device='cpu'):
    model.to(device)
    model.eval()

    desiredLabel = desiredLabel.to(device)

    TP, FP, FN, TN = 0, 0, 0, 0
    valid = []

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

                if validation:
                    valid.append((images_data[j], target_labels.data.max(dim=0)[1].type(torch.int64), predicted_labels[0]))

    accuracy = (TP + TN) / len(testloader)
    sensitivity = safe_division(TP, (TP+FN))  # WHAT WE WANT
    specificity = safe_division(TN, (TN+FP))  # WHAT WE WANT
    ppv = safe_division(TP, (TP+FP))
    npv = safe_division(TN, (TN+FN))

    f1 = safe_division((2 * (ppv * sensitivity)), (ppv + sensitivity))  # balance view of model

    if displayPrint:
        print(f"Total={len(testloader)}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
        print('Testing Accuracy: {:.3f}'.format(accuracy))
        print('Testing Sensitivity: {:.3f}'.format(sensitivity))
        print('Testing Specificity: {:.3f}'.format(specificity))
        print('Testing PPV: {:.3f}'.format(ppv))
        print('Testing NPV: {:.3f}'.format(npv))
        print('Testing F1 Score: {:.3f}'.format(f1))
        print('Confusion Matrix')
        print('          Predicted N | Predicted P')
        print('         ---------------------------')
        print('Ground N |   TN = {}  |   FP = {}   |').format(TN, FP)
        print('Ground P |   FN = {}  |   TP = {}   |').format(FN, TP)
        print('\n')

    return valid


# original test function
# model: the model to be tested
# testloader: containing the test data
# desiredLabel: tensor of the label you want to pass to the second classifier. this will also be the "positive"
# displayPrint: bool to tell if the print statements are to be shown or not
# validation: bool to return values into the validation set for display
def test_original(model, testloader, desiredLabel, displayPrint, validation, device='cpu'):
    model.to(device)
    model.eval()

    desiredLabel = desiredLabel.to(device)

    TP, FP, FN, TN = 0, 0, 0, 0

    valid = []

    with torch.no_grad():
        for batch_idx, (images_data, target_labels) in enumerate(testloader):
            images_data, target_labels = images_data.to(device), target_labels.to(device)
            images_data = transform(images_data)
            output = model(images_data)
            predicted_labels = torch.exp(output).max(dim=1)[1]

            for j in range(images_data.size()[0]):

                if torch.equal(predicted_labels[j], desiredLabel[0]):

                    # true positive
                    if torch.equal(target_labels.data.max(dim=1)[1], desiredLabel[0]):
                        TP += 1

                    # false positive
                    else:
                        FP += 1

                # negative
                else:

                    # false negative
                    if torch.equal(target_labels.data.max(dim=1)[1], desiredLabel[0]):
                        FN += 1

                    # true negative
                    else:
                        TN += 1

                if validation:
                    valid.append((images_data[j], target_labels.data.max(dim=1)[1].type(torch.int64), predicted_labels[0]))

    accuracy = (TP + TN) / len(testloader)
    sensitivity = safe_division(TP, (TP+FN))  # WHAT WE WANT
    specificity = safe_division(TN, (TN+FP))  # WHAT WE WANT
    ppv = safe_division(TP, (TP+FP))
    npv = safe_division(TN, (TN+FN))

    f1 = safe_division((2 * (ppv * sensitivity)), (ppv + sensitivity))  # balance view of model

    if displayPrint:
        print(f"Total={len(testloader)}, TP={TP}, FP={FP}, FN={FN}, TN={TN}")
        print('Testing Accuracy: {:.3f}'.format(accuracy))
        print('Testing Sensitivity: {:.3f}'.format(sensitivity))
        print('Testing Specificity: {:.3f}'.format(specificity))
        print('Testing PPV: {:.3f}'.format(ppv))
        print('Testing NPV: {:.3f}'.format(npv))
        print('Testing F1 Score: {:.3f}'.format(f1))
        print('Confusion Matrix')
        print('          Predicted N | Predicted P')
        print('         ---------------------------')
        print('Ground N |   TN = {}  |   FP = {}   |').format(TN, FP)
        print('Ground P |   FN = {}  |   TP = {}   |').format(FN, TP)
        print('\n')

    return valid


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


# independent validation normal dataset
def __get_binary_normal_valid_dataset(img_size, batch_size):
    class_dict = {0: 'normal', 1: 'infected'}
    val_groups = ['val']
    val_numbers = {'val_normal': 8,
                   'val_infected': 8,
                   }

    valset_paths = {'val_normal': './dataset/val/normal',
                    'val_infected': './dataset/val/infected/covid',
                    }

    valset1 = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    val_numbers = {'val_normal': 0,
                   'val_infected': 8,
                   }

    valset_paths = {'val_normal': './dataset/val/normal',
                    'val_infected': './dataset/val/infected/non-covid',
                    }

    valset2 = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    # load dataset
    valsets = ConcatDataset([valset1, valset2])
    validationloader = DataLoader(valsets, batch_size=batch_size, shuffle=True)

    return validationloader


# independent validation covid dataset
def __get_binary_covid_valid_dataset(img_size, batch_size):
    class_dict = {0: 'non-covid', 1: 'covid'}
    val_groups = ['val']
    val_numbers = {'val_non-covid': 8,
                   'val_covid': 8,
                   }

    valset_paths = {'val_non-covid': './dataset/val/infected/non-covid',
                    'val_covid': './dataset/val/infected/covid',
                    }

    valset = BinaryClassDataset('val', img_size, class_dict, val_groups, val_numbers, valset_paths)

    # load dataset
    validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    return validationloader


# magic inside
# adds a new set of labels to each test sample
def __get_binary_piped_valid_dataset(img_size, batch_size):
    class_dict = {0: 'normal', 1: 'infected'}
    groups = ['val']
    dataset_numbers = {'val_normal': 8,
                       'val_infected': 8,
                       }

    dataset_paths = {'val_normal': './dataset/val/normal/',
                     'val_infected': './dataset/val/infected/non-covid',
                     }

    # normal, infected
    # normal, non covid, covid
    valset1 = BinaryClassDataset('val', img_size, class_dict, groups, dataset_numbers, dataset_paths)

    val1 = []
    # appends an additional pair of labels
    # first pair: [normal, infected]
    # second pair: [normal, non-covid, covid] --> hardcoded tensor
    for i in range(len(valset1)):
        if torch.equal(valset1[i][1], torch.tensor([1., 0.])):
            val1.append((valset1[i][0], valset1[i][1], torch.tensor([1., 0., 0.])))
        elif torch.equal(valset1[i][1], torch.tensor([0., 1.])):
            val1.append((valset1[i][0], valset1[i][1], torch.tensor([0., 1., 0.])))

    dataset_numbers = {'val_normal': 0,
                       'val_infected': 8,
                       }

    dataset_paths = {'val_normal': './dataset/val/normal/',
                     'val_infected': './dataset/val/infected/covid',
                     }

    valset2 = BinaryClassDataset('val', img_size, class_dict, groups, dataset_numbers, dataset_paths)

    val2 = []
    # appends an additional pair of labels
    # first pair: [normal, infected]
    # second pair: [normal, non-covid, covid] --> hardcoded tensor
    for i in range(len(valset2)):
        if torch.equal(valset2[i][1], torch.tensor([0., 1.])):
            val2.append((valset2[i][0], valset2[i][1], torch.tensor([0., 0., 1.])))

    valsetNormal = ConcatDataset([val1, val2])
    valloaderNormal = DataLoader(valsetNormal, batch_size=batch_size, shuffle=True)

    return valloaderNormal


def __get_label(label, labelOrder):
    return labelOrder[label]


def __display_validation(valset1, valset2, isIndependent, pos, device="cpu"):

    covid, noncovid, normal, infected = [], [], [], []
    pos = pos.to(device)

    if not isIndependent:

        # valset1: only normal
        # valset2: mix of covid and noncovid
        for i in range(len(valset2)):
            if torch.equal(valset2[i][1], pos):
                covid.append(valset2[i])
            else:
                noncovid.append(valset2[i])

        print("displaying validation piped results in the format of (target) -> (predicted)")

        # show all in same diagram
        f, (r1, r2, r3) = plt.subplots(3, max(max(len(covid), len(noncovid)), len(valset1)), squeeze=False, figsize=(20, 6))
        for j in range(len(valset1)):
            r1[j].imshow(valset1[j][0][0])
            r1[j].axis('off')
            r1[j].set_title("{} -> {}".format(__get_label(valset1[j][1][0], ['n', 'i']), __get_label(valset1[j][2][0], ['n', 'i']), fontsize=4))

        for l in range(len(noncovid)):
            r2[l].imshow(noncovid[l][0][0])
            r2[l].axis('off')
            r2[l].set_title("{} -> {}".format(__get_label(noncovid[l][1], ['nc', 'c']), __get_label(noncovid[l][2], ['nc', 'c']), fontsize=4))

        for k in range(len(covid)):
            r3[k].imshow(covid[k][0][0])
            r3[k].axis('off')
            r3[k].set_title("{} -> {}".format(__get_label(covid[k][1], ['nc', 'c']), __get_label(covid[k][2], ['nc', 'c']), fontsize=4))

        # tidying
        if len(covid) >= len(noncovid) and len(covid) >= len(valset1):
            for i in range(len(noncovid), len(covid), 1):
                f.delaxes(r2[i])
                # r2[i].set_visible(False)
            for j in range(len(valset1), len(covid), 1):
                f.delaxes(r1[j])
                # r1[j].set_visible(False)
        elif len(noncovid) >= len(covid) and len(noncovid) >= len(valset1):
            for i in range(len(covid), len(noncovid), 1):
                f.delaxes(r3[i])
                # r3[i].set_visible(False)
            for j in range(len(valset1), len(noncovid), 1):
                f.delaxes(r1[j])
                # r1[j].set_visible(False)
        elif len(valset1) >= len(covid) and len(valset1) >= len(noncovid):
            for i in range(len(covid), len(valset1), 1):
                f.delaxes(r3[i])
                # r3[i].set_visible(False)
            for j in range(len(noncovid), len(valset1), 1):
                f.delaxes(r2[j])
                # r2[j].set_visible(False)

    else:
        # valset1: mix of normal and infected
        for i in range(len(valset1)):
            if torch.equal(valset1[i][1], pos):
                infected.append(valset1[i])
            else:
                normal.append(valset1[i])

        # valset2: mix of covid and noncovid
        for j in range(len(valset2)):
            if torch.equal(valset2[j][1], pos):
                covid.append(valset2[j])
            else:
                noncovid.append(valset2[j])

        # plot for normal vs infected validation
        print("displaying normal validation independent results in the format of (target) -> (predicted)")
        f, (r1, r2) = plt.subplots(2, max(len(normal), len(infected)), squeeze=False, figsize=(15, 4))
        for j in range(len(normal)):
            r1[j].imshow(normal[j][0][0], interpolation='nearest')
            r1[j].axis('off')
            r1[j].set_title("{} -> {}".format(__get_label(normal[j][1][0], ["n", "i"]), __get_label(normal[j][2], ["n", "i"]), fontsize=4))

        for k in range(len(infected)):
            r2[k].imshow(infected[k][0][0])
            r2[k].axis('off')
            r2[k].set_title("{} -> {}".format(__get_label(infected[j][1][0], ["nc", "c"]), __get_label(normal[j][2], ["nc", "c"]), fontsize=4))

        # tidying first figure
        if len(normal) < len(infected):
            for i in range(len(normal), len(infected), 1):
                f.delaxes(r1[i])
                # r1[i].set_visible(False)
        elif len(normal) > len(infected):
            for j in range(len(infected), len(normal), 1):
                f.delaxes(r2[j])
#                 r2[j].set_visible(False)

        # plot for covid vs noncovid validation
        print("displaying covid validation independent results in the format of (target) -> (predicted)")
        f, (r3, r4) = plt.subplots(2, max(len(covid), len(noncovid)), squeeze=False, figsize=(15, 4))
        for m in range(len(covid)):
            r3[m].imshow(covid[m][0][0])
            r3[m].axis('off')
            r3[m].set_title("{} -> {}".format(covid[m][1][0], covid[m][2]), fontsize=10)

        for l in range(len(noncovid)):
            r4[l].imshow(noncovid[l][0][0])
            r4[l].axis('off')
            r4[l].set_title("{} -> {}".format(noncovid[l][1][0], covid[l][2]), fontsize=10)

        # tidying second figure
        if len(covid) < len(noncovid):
            for i in range(len(covid), len(noncovid), 1):
                f.delaxes(r3[i])
                # r3[i].set_visible(False)
        elif len(covid) > len(noncovid):
            for j in range(len(noncovid), len(covid), 1):
                f.delaxes(r4[j])
                # r4[j].set_visible(False)

    # Display full plot
    plt.show()


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


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="test image classifier model")
    parser.add_argument("--independent", type=bool, default=True, help="true for independent, false for piped")
    parser.add_argument("--validation", type=int, default=False, help="false for no validation, true for validation")
    parser.add_argument("--print", type=bool, default=True, help="true to print stats, false otherwise")
    parser.add_argument("--output_var", type=int, default=2, help="number of output variables")
    parser.add_argument("--batch_size", type=int, default=1, help="set testing batch size")
    parser.add_argument("--gpu", action="store_const", const="cuda", default="cuda", help="use gpu")
    parser.add_argument("--normalclf", type=str, default="binaryModelNormalBestSensitivity", help="input model name to be used as normal classifier")
    parser.add_argument("--covidclf", type=str, default="binaryModelCovidBestSensitivity", help="input model name to be used as covid classifier")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    output_var = args.output_var
    validation = args.validation
    independent = args.independent
    batch_size = args.batch_size
    displayPrint = args.display_print

    # set and load dataset spec
    img_size = (150, 150)

    # model parameters
    covidCLFPath = 'models/' + args.covidclf
    normalCLFPath = 'models/' + args.normalclf

    # doing binary classifier
    if output_var == 2:

        model = RestNet(2)
        pos = torch.tensor([1]).type(torch.int64)

        if validation and independent:
            print("Starting: Validation set on Normal Independent classifier")

            normalValidTestLoader = __get_binary_normal_valid_dataset(img_size, batch_size)

            model.load_state_dict(torch.load(normalCLFPath, map_location=torch.device('cpu')))
            nvs1 = test_original(model, normalValidTestLoader, pos, displayPrint, validation) # normal vs infected

            print("Starting: Validation set on Covid Independent classifier")

            covidValidTestLoader = __get_binary_covid_valid_dataset(img_size, batch_size)
            model.load_state_dict(torch.load(covidCLFPath, map_location=torch.device('cpu')))
            nvs2 = test_original(model, covidValidTestLoader, pos, displayPrint, validation)

            __display_validation(nvs1, nvs2, independent, pos)

        elif validation and not independent:
            print("Starting: Validation set on Normal Piped classifier")

            normalPipedTestLoader = __get_binary_piped_valid_dataset(img_size, batch_size)
            model.load_state_dict(torch.load(normalCLFPath, map_location=torch.device('cpu')))
            intermediate, nvs1 = test_first_binary(model, normalPipedTestLoader, pos, displayPrint, validation)

            print("Starting: Validation set on Covid Piped classifier")
            model.load_state_dict(torch.load(covidCLFPath, map_location=torch.device('cpu')))
            nvs2 = test_second_binary(model, intermediate, pos, displayPrint, validation)

            __display_validation(nvs1, nvs2, independent, pos)

        elif not validation and independent:
            print("Starting: Test set on Normal Independent classifier")

            normalIndependentTestLoader = __get_binary_normal_test_dataset(img_size, batch_size)
            model.load_state_dict(torch.load(normalCLFPath, map_location=torch.device('cpu')))
            unused = test_original(model, normalIndependentTestLoader, pos, displayPrint, validation)

            print("Starting Test set on Covid Independent classifier")

            covidIndependentTestLoader = __get_binary_covid_test_dataset(img_size, batch_size)
            model.load_state_dict(torch.load(covidCLFPath, map_location=torch.device('cpu')))
            unused = test_original(model, covidIndependentTestLoader, pos, displayPrint, validation)

        else:
            print("Starting: Test set on Normal Piped classifier")

            normalPipedTestLoader = __get_binary_piped_test_dataset(img_size, batch_size)
            model.load_state_dict(torch.load(normalCLFPath, map_location=torch.device('cpu')))
            intermediate, unused = test_first_binary(model, normalPipedTestLoader, pos, displayPrint, validation)

            print("Starting: Test set on Covid Piped classifier")
            model.load_state_dict(torch.load(covidCLFPath, map_location=torch.device('cpu')))
            unused = test_second_binary(model, intermediate, pos, displayPrint, validation)


    else:
        print("output_var must be 2")

