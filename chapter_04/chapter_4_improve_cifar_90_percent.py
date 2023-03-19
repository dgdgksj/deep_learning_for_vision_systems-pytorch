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

class Simple_cnn_v2(nn.Module):
	# pp. 233 ~ 235
	def __init__(self):
		super().__init__()
		# CONV!
		self.layer1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=32
			),
			nn.ReLU(),
			# nn.MaxPool2d(2),
			nn.BatchNorm2d(num_features=32) # 케라스 배치정규화 파라미터 수가 파이토치랑 다른데 링크 보면 의문이 해결될듯 https://stackoverflow.com/questions/60079783/difference-between-keras-batchnormalization-and-pytorchs-batchnorm2d
		)
		# CONV2
		self.layer2 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=32
			),
			nn.ReLU(),
			nn.BatchNorm2d(num_features=32),
		)
		# POOL + Dropout
		self.layer3 = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Dropout(0.2),
		)
		# CONV3
		self.layer4 = nn.Sequential(
			nn.Conv2d(
				in_channels=32,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=64
			),
			nn.ReLU(),
			nn.BatchNorm2d(num_features=64)
		)
		#CONV4
		self.layer5 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=64
			),
			nn.ReLU(),
			nn.BatchNorm2d(num_features=64)
		)
		# POOL + Dropout
		self.layer6 = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Dropout(0.3),
		)
		# CONV5
		self.layer7 = nn.Sequential(
			nn.Conv2d(
				in_channels=64,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=128
			),
			nn.ReLU(),
			nn.BatchNorm2d(num_features=128)
		)
		# CONV6
		self.layer8 = nn.Sequential(
			nn.Conv2d(
				in_channels=128,
				kernel_size=(3, 3),
				stride=1,
				padding='same',
				out_channels=128
			),
			nn.ReLU(),
			nn.BatchNorm2d(num_features=128)
		)
		# POOL + Dropout
		self.layer9 = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Dropout(0.4),
		)
		# FC7
		self.layer10 = nn.Sequential(
			nn.Flatten(1),
			# 4 * 4 * 128
			nn.Linear(in_features=2048, out_features=10),
			nn.Softmax()
		)

	def forward(self, inputs):
		x = inputs
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		x = self.layer9(x)
		x = self.layer10(x)
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
	# optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9,weight_decay=opt.weight_decay)
	optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
	loss_avg = Averager()
	transform = transforms.Compose(
		[
			transforms.Resize((32, 32)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
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
	best_accuracy = -1
	while(True):
		model.train()
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
		if (iteration + 1) % opt.valInterval == 0 or iteration == 0:  # To see training progress, we also conduct validation when 'iteration == 0'
			correct = 0
			total = 0

			model.eval()
			train_loss = loss_avg.val()
			loss_avg.reset()
			with torch.no_grad():
				for inputs, labels  in validloader:
					inputs = inputs.to(device)
					labels = labels.to(device)

					# 신경망에 이미지를 통과시켜 출력을 계산합니다
					outputs = model(inputs)
					loss = criterion(outputs, labels)
					loss_avg.add(loss)
					# 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
			elapsed_time = time.time() - start_time
			current_accuracy = 100 * correct // total
			if current_accuracy > best_accuracy:
				best_accuracy = current_accuracy
				torch.save(model.state_dict(), os.path.join(os.path.join(opt.model_save_path,opt.exp_name),"best_accuracy_"+str(100 * correct // total)+".pth"))
			loss_log = f'[{iteration + 1}/{opt.num_iter}] train loss: {train_loss:0.5f}, valid loss: {loss_avg.val():0.5f}, Accuracy: {100 * correct // total}%, Elapsed_time: {elapsed_time:0.5f}'
			loss_avg.reset()
			print(loss_log)
			model.train()
		if (iteration + 1) == opt.num_iter:
			print('end the training')
			sys.exit()
		iteration += 1

if __name__ == '__main__':
	opt = argparse.Namespace(
		batch_size=int(128),
		exp_name = str("cifar10_chapter4"),
		workers=int(4),
		num_iter=int(125),
		weight_decay = 1e-6,
		valInterval=int(1),
		lr=1e-4,
		model_save_path=str('../saved_model'),
		class_names=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	)
	if not os.path.exists(os.path.join(opt.model_save_path,opt.exp_name)):
		os.makedirs(os.path.join(opt.model_save_path,opt.exp_name),exist_ok=True)
	model = Simple_cnn_v2()
	model.cuda()
	summary(model,input_size=(3, 32, 32))
	train(opt,model)
