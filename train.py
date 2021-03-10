# Define classifier class
class NN_Classifier(nn.Module):
	def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
		''' Builds a feedforward network with arbitrary hidden layers.

		    Arguments
		    ---------
		    input_size: integer, size of the input
		    output_size: integer, size of the output layer
		    hidden_layers: list of integers, the sizes of the hidden layers
		    drop_p: float between 0 and 1, dropout probability
		'''
		super().__init__()
		# Add the first layer, input to a hidden layer
		self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

		# Add a variable number of more hidden layers
		layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
		self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

		self.output = nn.Linear(hidden_layers[-1], output_size)

		self.dropout = nn.Dropout(p=drop_p)

	def forward(self, x):
		''' Forward pass through the network, returns the output logits '''

		# Forward through each layer in `hidden_layers`, with ReLU activation and dropout
		for linear in self.hidden_layers:
			x = F.relu(linear(x))
			x = self.dropout(x)

		x = self.output(x)

		return F.log_softmax(x, dim=1)

def validate(model, validloader, criterion, device='cuda'):
	model.to(device)
	model.eval()
	test_loss = 0
	accuracy = 0

	for batch_idx, (images_data, target_labels) in enumerate(validloader):
		images_data, target_labels = images_data.to(device), target_labels.to(device)
		output = model(images_data)

		test_loss += criterion(output, target_labels).item()

		predicted_labels = torch.exp(output).max(dim=1)[1]

		equality = (target_labels.data.max(dim=1)[1] == predicted_labels)
		accuracy += equality.type(torch.FloatTensor).mean()

	return test_loss, accuracy

def train(model, train_loader, optimiser, epochs, path, saving):
	model = getattr(models, model_name)(pretrained=True)

	# Freeze parameters that we don't need to re-train
	for param in model.parameters():
		param.requires_grad = False

	# Make classifier
	n_in = next(model.classifier.modules()).in_features
	n_out = len(labelsdict)
	model.classifier = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)

	# Define criterion and optimizer
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

	model.to(device)
	start = time.time()

	epochs = n_epoch
	steps = 0
	running_loss = 0
	print_every = 40
	training_losses = [] ### RECORD TRAINING LOSSES
	validation_losses = [] ### RECORD VALIDATION LOSSES
	for e in range(epochs):
	    model.train()
	    for images, labels in trainloader:
	        images, labels = images.to(device), labels.to(device)

	        steps += 1

	        optimizer.zero_grad()

	        output = model.forward(images)
	        loss = criterion(output, labels)
	        loss.backward()
	        optimizer.step()

	        running_loss += loss.item()

	        if steps % print_every == 0:
	            # Eval mode for predictions
	            model.eval()

	            # Turn off gradients for validation
	            with torch.no_grad():
	                test_loss, accuracy = validate(model, validloader, criterion, device)
	                training_losses.append(running_loss/print_every) ### RECORD TRAINING LOSSES
	                validation_losses.append(test_loss/len(validloader)) ### RECORD VALIDATION LOSSES

	            print("Epoch: {}/{} - ".format(e+1, epochs),
	                  "Training Loss: {:.3f} - ".format(running_loss/print_every),
	                  "Validation Loss: {:.3f} - ".format(test_loss/len(validloader)),
	                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

	            running_loss = 0

	            # Make sure training is back on
	            model.train()

	print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
	print(f"Run time: {(time.time() - start)/60:.3f} min")
	return model, training_losses, validation_losses
