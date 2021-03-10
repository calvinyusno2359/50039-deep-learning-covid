# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

# All images are of size 150 x 150
size = (150, 150)

# Only two classes will be considered here (normal and infected)
classes = {0: 'normal', 1: 'infected'}
print(classes)

# The dataset has been split in training, testing and validation groups
groups = ['train', 'test', 'val']
print(groups)

# Number of images in each part of the dataset
dataset_numbers = {'train_normal': 36,\
                   'train_infected': 34,\
                   'val_normal': 4,\
                   'val_infected': 4,\
                   'test_normal': 14,\
                   'test_infected': 13}
print(dataset_numbers)

# Path to images for different parts of the dataset
dataset_paths = {'train_normal': './dataset_demo/train/normal/',\
                 'train_infected': './dataset_demo/train/infected/',\
                 'val_normal': './dataset_demo/val/normal/',\
                 'val_infected': './dataset_demo/val/infected/',\
                 'test_normal': './dataset_demo/test/normal/',\
                 'test_infected': './dataset_demo/test/infected/'}
print(dataset_paths)

# Display an image
path_to_file = './dataset_demo/train/normal/1.jpg'
with open(path_to_file, 'rb') as f:
    im = np.asarray(Image.open(f))
    plt.imshow(im)
f.close()

# Image shape is indeed 150 x 150
print(im.shape)

# Images are defined as a Numpy array of values between 0 and 256
print(im)

class Lung_Dataset(Dataset):
    """
    Generic Dataset class.
    """

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected'}

        # The dataset has been split in training, testing and validation datasets
        self.groups = ['train', 'test', 'val']

        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': 36,\
                                'train_infected': 34,\
                                'val_normal': 4,\
                                'val_infected': 4,\
                                'test_normal': 14,\
                                'test_infected': 13}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': './dataset_demo/train/normal/',\
                              'train_infected': './dataset_demo/train/infected/',\
                              'val_normal': './dataset_demo/val/normal/',\
                              'val_infected': './dataset_demo/val/infected/',\
                              'test_normal': './dataset_demo/test/normal/',\
                              'test_infected': './dataset_demo/test/infected/'}


    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the Lung Dataset used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "Images have been split in three groups: training, testing and validation sets.\n"
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)


    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            # Convert to Numpy array and normalize pixel values by dividing by 255.
            im = np.asarray(Image.open(f))/255
        f.close()
        return im


    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)

ld = Lung_Dataset()
ld.describe()

im = ld.open_img('train', 'normal', 1)
print(im.shape)
print(im)

ld.show_img('train', 'normal', 1)

class Lung_Train_Dataset(Dataset):

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected'}

        # The dataset consists only of training images
        self.groups = 'train'

        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': 36,\
                                'train_infected': 34}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': './dataset_demo/train/normal/',\
                              'train_infected': './dataset_demo/train/infected/'}


    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the training dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)


    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im


    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)


    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())


    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """

        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0])
        else:
            class_val = 'infected'
            index = index - first_val
            label = torch.Tensor([0, 1])
        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label

ld_train = Lung_Train_Dataset()
ld_train.describe()

print(len(ld_train))

im, class_oh = ld_train[64]
print(im.shape)
print(im)
print(class_oh)

class Lung_Test_Dataset(Dataset):

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected'}

        # The dataset consists only of test images
        self.groups = 'test'

        # Number of images in each part of the dataset
        self.dataset_numbers = {'test_normal': 14,\
                                'test_infected': 13}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'test_normal': './dataset_demo/test/normal/',\
                              'test_infected': './dataset_demo/test/infected/'}


    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the test dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)


    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im


    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)


    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())


    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """

        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0])
        else:
            class_val = 'infected'
            index = index - first_val
            label = torch.Tensor([0, 1])
        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label

ld_test = Lung_Test_Dataset()
ld_test.describe()

print(len(ld_test))

im, class_oh = ld_test[18]
print(im.shape)
print(im)
print(class_oh)

class Lung_Val_Dataset(Dataset):

    def __init__(self):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected'}

        # The dataset consists only of validation images
        self.groups = 'val'

        # Number of images in each part of the dataset
        self.dataset_numbers = {'val_normal': 4,\
                                'val_infected': 4}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'val_normal': './dataset_demo/val/normal/',\
                              'val_infected': './dataset_demo/val/infected/'}


    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the validation dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)


    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im


    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)


    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())


    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """

        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0])
        else:
            class_val = 'infected'
            index = index - first_val
            label = torch.Tensor([0, 1])
        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label

ld_val = Lung_Val_Dataset()
ld_val.describe()

print(len(ld_val))

im, class_oh = ld_val[3]
print(im.shape)
print(im)
print(class_oh)

# Batch size value to be used (to be decided freely, but set to 4 for demo)
bs_val = 4

# Dataloader from dataset (train)
train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)
print(train_loader)

# Dataloader from dataset (test and val)
test_loader = DataLoader(ld_test, batch_size = bs_val, shuffle = True)
print(test_loader)
val_loader = DataLoader(ld_val, batch_size = bs_val, shuffle = True)
print(val_loader)

# Typical mini-batch for loop on dataloader (train)
for k, v in enumerate(train_loader):
    print("-----")
    print(k)
    print(v[0])
    print(v[1])
    # Forced stop
    break
    #assert False, "Forced stop after one iteration of the for loop"

