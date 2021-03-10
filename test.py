import argparse

from dataset import Image_Dataset_Part
from torch.utils.data import DataLoader

def predict(datapath, model, topk, device):
	model.to(device)
	model.eval()

	img = process_image(image_path)
	img = img.to(device)

	output = model.forward(img)
	ps = torch.exp(output)
	probs, idxs = ps.topk(topk)

	idx_to_class = dict((v,k) for k, v in model.classifier.class_to_idx.items())
	classes = [v for k, v in idx_to_class.items() if k in idxs.to('cpu').numpy()]

	if cat_to_name:
	    classes = [cat_to_name[str(i + 1)] for c, i in \
	                 model.classifier.class_to_idx.items() if c in classes]

	print('Probabilities:', probs.data.cpu().numpy()[0].tolist())
	print('Classes:', classes)

if __name__ == "__main__":
	img_size = (150, 150)
	class_dict = {0: 'normal', 1: 'infected'}
	groups = ['test']
	dataset_numbers = { 'test_normal': 36,
											'test_infected': 34,
										}

	dataset_paths = { 'test_normal': './dataset_demo/test/normal/',
										'test_infected': './dataset_demo/test/infected/',
									}

	bs_val = 4
	testset = Image_Dataset_Part('test', img_size, class_dict, groups, dataset_numbers, dataset_paths)
	test_loader = DataLoader(testset, batch_size = bs_val, shuffle = True)

	# Try model on one mini-batch
	for batch_idx, (images_data, target_labels) in enumerate(test_loader):
	    predicted_labels = model(images_data)
	    print(predicted_labels)
	    print(target_labels)
	    # Forced stop
	    break
	    #assert False, "Forced stop after one iteration of the mini-batch for loop"

	print("ok")
