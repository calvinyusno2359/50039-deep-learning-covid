import json
import torch
from PIL import Image
from collections import OrderedDict
from torch import nn, optim
import torch.nn.functional as F


def save_model(model, path):
    checkpoint = {'c_input': model.fc1.in_features,
                  'c_out': model.fc1.out_features}
    torch.save(checkpoint, path)


def load_model(model, path):
    cp = torch.load(path)
    model.fc1.in_features = cp['c_input']
    model.fc1.out_features = cp['c_out']
    return model
