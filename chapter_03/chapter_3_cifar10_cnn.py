# import keras.utils
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset
# from keras.layers import MaxPool2D
# from torchinfo import summary
from torchsummary import summary

# from utils import Averager
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import sys
sys.path.append(os.path.dirname(os.path.abspath((os.path.dirname(__file__)))))
from utils import Averager
import torch.nn.functional as F

class Simple_perceptron(nn.Module):
	# pp. 129 ~ 131
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Sequential(
			nn.Flatten(start_dim=3),
			nn.Linear(in_features=784, out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=10),
			nn.Softmax(10)
		)

	def forward(self, inputs):
		x = inputs
		return self.layer1(x)


class Simple_cnn_v1(nn.Module):
	# pp. 157 ~ 158
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=32
			),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=64
			),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer3 = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.3),
			nn.Linear(in_features=3136, out_features=64),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(in_features=64, out_features=10),
			nn.Softmax()
		)

	# self.fc1 = nn.Linear(10 * 12 * 12, 50)
	def forward(self, inputs):
		x = inputs
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		return x

class Simple_cnn_v2(nn.Module):
	# pp. 178 ~ 179
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				kernel_size=(2, 2),
				stride=1,
				padding='same',
				out_channels=16
			),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(
				in_channels=16,
				kernel_size=(2, 2),
				stride=1,
				padding='same',
				out_channels=32
			),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				kernel_size=(2, 2),
				stride=1,
				padding='same',
				out_channels=64
			),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer4 = nn.Sequential(
			nn.Dropout(0.3),
			nn.Flatten(1),
			nn.Linear(in_features=1024, out_features=500),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(in_features=500, out_features=10),
			# nn.Softmax()
		)

	# self.fc1 = nn.Linear(10 * 12 * 12, 50)
	def forward(self, inputs):
		x = inputs
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x

def imshow(img):
	# print(img.shape,type(img))
	img = img / 2 + 0.5  # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()
def train(opt,model):
	criterion = nn.CrossEntropyLoss()
	# optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr, momentum=0.9)
	optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
	loss_avg = Averager()
	transform = transforms.Compose(
		[
			transforms.Resize((32, 32)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)
	trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
	train_idx, valid_idx = train_test_split(list(range(len(trainset))), test_size=0.2)
	trainset, validset = Subset(trainset, train_idx), Subset(trainset, valid_idx)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
	                                          num_workers=opt.workers)
	validloader = torch.utils.data.DataLoader(validset, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
	                                          num_workers=opt.workers)

	testset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
	start_iter = 0
	iteration = start_iter
	start_time = time.time()
	while(True):
		model.train()
		cnt = 0
		for inputs, labels in trainloader:
			optimizer.zero_grad()
			# imshow(torchvision.utils.make_grid(inputs))
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.requires_grad_(True)
			loss.backward()
			optimizer.step()
			loss_avg.add(loss)
		# break
		elapsed_time = time.time() - start_time
		loss_log = f'[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f},Elapsed_time: {elapsed_time:0.5f}'
		loss_avg.reset()
		print(loss_log)
		# if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
		# 	model.eval()
		iteration += 1


	# for epoch in tqdm(range(num_epochs)):
	# 	for phase in ['train', 'valid']:
	# 		if phase == 'train':
	# 			cnn.train()  # Set model to training mode
	# 		else:
	# 			cnn.eval()  # Set model to evaluate mode
	#
	# 		running_loss, running_corrects, num_cnt = 0.0, 0, 0
	# pass

if __name__ == '__main__':
	opt = argparse.Namespace(
		batch_size=int(32),
		workers=int(4),
		num_iter=int(200),
		valInterval=int(10),
		lr=float(0.001),
		saved_model=str('./saved_model'),
		class_names=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	)
	model = Simple_cnn_v2()
	model.cuda()
	summary(model,input_size=(3, 32, 32))
	train(opt,model)

	# transform = transforms.Compose(
	# 	[transforms.Resize((32, 32)),
	# 		transforms.ToTensor(),
	# 	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	# batch_size = 16
	# num_workers = 2
	# trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True,transform=transform)
	# train_idx, valid_idx = train_test_split(list(range(len(trainset))), test_size=0.2)
	# trainset, validset = Subset(trainset, train_idx), Subset(trainset, valid_idx)
	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
	# validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
	#
	# testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True,transform=transform)
	# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
	# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	# dataiter = iter(trainloader)
	# images, labels = next(dataiter)
	# #
	# # 이미지 보여주기
	# imshow(torchvision.utils.make_grid(images))
	# # 정답(label) 출력
	# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	# print(len(trainset))
	# print(len(validset))