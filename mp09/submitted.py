# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


"""
1. Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
	def __init__(self, data_files, transform=None, target_transform=None):
		"""
		Initialize your dataset here. Note that transform and target_transform
		correspond to your data transformations for train and test respectively.
		"""
		self.data = []
		self.labels = []
		for filename in data_files:
			data_dict = unpickle(filename)
			self.data.extend(data_dict[b'data'])
			self.labels.extend(data_dict[b'labels'])
		for i in range(len(self.data)):
			self.data[i] = np.reshape(self.data[i], (3, 32, 32))
			self.data[i] = np.transpose(self.data[i], (1, 2, 0))
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		"""
		Return the length of your dataset here.
		"""
		return len(self.labels)

	def __getitem__(self, idx):
		"""
		Obtain a sample from your dataset.

		Parameters:
			x:	an integer, used to index into your data.

		Outputs:
			y:	a tuple (image, label), although this is arbitrary so you can use whatever you would like.
		"""
		img = self.data[idx]
		label = self.labels[idx]
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			label = self.target_transform(label)
		return img, label


def get_preprocess_transform(mode):
	"""
	Parameters:
		mode:		"train" or "test" mode to obtain the corresponding transform
	Outputs:
		transform:	a torchvision transforms object e.g. transforms.Compose([...]) etc.
	"""
	if mode == "train":
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		])
	elif mode == "test":
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		])
	else:
		raise ValueError(f"Invalid mode: {mode}")
	return transform


def build_dataset(data_files, transform=None):
	"""
	Parameters:
		data_files:	a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
		transform:	the preprocessing transform to be used when loading a dataset sample
	Outputs:
		dataset:	a PyTorch dataset object to be used in training/testing
	"""
	dataset = CIFAR10(data_files, transform=transform)
	return dataset


"""
2. Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
	"""
	Parameters:
		dataset:		a PyTorch dataset to load data
		loader_params:	a dict containing all the parameters for the loader.

	Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those
	respective parameters in the PyTorch DataLoader class.

	Outputs:
		dataloader:		a PyTorch dataloader object to be used in training/testing
	"""
	dataloader = DataLoader(dataset, **loader_params)
	return dataloader


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
	def __init__(self, num_classes=10):
		"""
		Initialize your neural network here. Remember that you will be performing finetuning
		in this network so follow these steps:

		1. Initialize convolutional backbone with pretrained model parameters.
		2. Freeze convolutional backbone.
		3. Initialize linear layer(s).
		"""
		super().__init__()
		################# Your Code Starts Here #################
		self.backbone = resnet18(pretrained=True)
		self.backbone.load_state_dict(torch.load('resnet18.pt'))
		for param in self.backbone.parameters():
			param.requires_grad = False
		self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
		################## Your Code Ends here ##################

	def forward(self, x):
		"""
		Perform a forward pass through your neural net.

		Parameters:
			x:	an (N, input_size) tensor, where N is arbitrary.

		Outputs:
			y:	an (N, output_size) tensor of output from the network
		"""
		################# Your Code Starts Here #################
		x = self.backbone(x)
		return x
		################## Your Code Ends here ##################


"""
3. (b) Build a model
"""
def build_model(trained=False):
	"""
	Parameters:
		trained:a bool value specifying whether to use a model checkpoint

	Outputs:
		model:		the model to be used for training/testing
	"""
	net = FinetuneNet()
	return net


"""
4. Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
	"""
	Parameters:
		optim_type:		the optimizer type e.g. "Adam" or "SGD"
		model_params:	the model parameters to be optimized
		hparams:		the hyperparameters (dict type) for usage with learning rate

	Outputs:
		optimizer:		a PyTorch optimizer object to be used in training
	"""
	if optim_type == "Adam":
		optimizer = torch.optim.Adam(model_params, lr=hparams["lr"])
	elif optim_type == "SGD":
		optimizer = torch.optim.SGD(model_params, lr=hparams["lr"], momentum=hparams["momentum"])
	else:
		raise ValueError(f"Invalid optimizer type: {optim_type}")
	return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
	"""
	Train your neural network.

	Iterate over all the batches in dataloader:
		1. The model makes a prediction.
		2. Calculate the error in the prediction (loss).
		3. Zero the gradients of the optimizer.
		4. Perform backpropagation on the loss.
		5. Step the optimizer.

	Parameters:
		train_dataloader:	a dataloader for the training set and labels
		model:				the model to be trained
		loss_fn:			loss function
		optimizer:			optimizer
	"""

	################# Your Code Starts Here #################
	for data, label in train_dataloader:
		optimizer.zero_grad()
		outputs = model(data)
		loss = loss_fn(outputs, label)
		loss.backward()
		optimizer.step()
	################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
	"""
	This part is optional.

	You can write this part to monitor your model training process.

	Test your neural network.
		1. Make sure gradient tracking is off, since testing set should only
			reflect the accuracy of your model and should not update your model.
		2. The model makes a prediction.
		3. Calculate the error in the prediction (loss).
		4. Print the loss.

	Parameters:
		test_dataloader:	a dataloader for the testing set and labels
		model:				the model that you will use to make predictions


	Outputs:
		test_acc:			the output test accuracy (0.0 <= acc <= 1.0)
	"""
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in test_dataloader:
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	accuracy = 100 * correct / total
	print('Test accuracy: {:.2f}%'.format(accuracy))

"""
7. Full model training and testing
"""
def run_model():
	"""
	The autograder will call this function and measure the accuracy of the returned model.
	Make sure you understand what this function does.
	Do not modify the signature of this function (names and parameters).

	Please run your full model training and testing within this function.

	Outputs:
		model:	trained model
	"""
	# Load hyperparameters
	hparams = {'optimizer': 'Adam', 'lr': 0.001, 'momentum': 0.9, 'batch_size': 64, 'epochs': 3, 'train_loader_params': {'batch_size': 64, 'shuffle': True}, 'test_loader_params': {'batch_size': 64, 'shuffle': False}}

	# Build PyTorch dataset
	train_dataset = build_dataset(['cifar10_batches/data_batch_1', 'cifar10_batches/data_batch_2', 'cifar10_batches/data_batch_3', 'cifar10_batches/data_batch_4', 'cifar10_batches/data_batch_5'], get_preprocess_transform('train'))
	test_dataset = build_dataset(['cifar10_batches/test_batch'], get_preprocess_transform('test'))

	# Build PyTorch dataloader
	train_dataloader = build_dataloader(train_dataset, hparams["train_loader_params"])
	test_dataloader = build_dataloader(test_dataset, hparams["test_loader_params"])

	# Build model
	model = build_model()

	# Build optimizer
	optimizer = build_optimizer(hparams["optimizer"], model.backbone.fc.parameters(), hparams)

	# Build loss function
	loss_fn = torch.nn.CrossEntropyLoss()

	# Train model
	for epoch in range(hparams["epochs"]):
		print(f"Epoch {epoch + 1} of {hparams['epochs']}")
		train(train_dataloader, model, loss_fn, optimizer)
		test(test_dataloader, model)

	return model
