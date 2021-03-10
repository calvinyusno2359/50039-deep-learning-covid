# A simple mode
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.fc1 = nn.Linear(87616, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim = 1)
        return output

# Create model
model = Net()

# Try model on one mini-batch
for batch_idx, (images_data, target_labels) in enumerate(train_loader):
    predicted_labels = model(images_data)
    print(predicted_labels)
    print(target_labels)
    # Forced stop
    break
    #assert False, "Forced stop after one iteration of the mini-batch for loop"
