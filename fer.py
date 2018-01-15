import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import argparse
import time

from network import Net
from datasets import FER2013Dataset



parser = argparse.ArgumentParser(description='PyTorch FER2013')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--GPU', type=int, default=0, metavar='N',
                    help='mode (default: CPU)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')



args = parser.parse_args()



batch_size = args.batch_size
data_transform = transforms.Compose([
	#transforms.Scale(64,64),
	transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
trainset = FER2013Dataset('/beegfs/jn1664/fer2013', train=True, transform=data_transform)
testset = FER2013Dataset('/beegfs/jn1664/fer2013', train=False, transform=data_transform)
#trainset = dsets.MNIST(root='.', train=True, download=True, transform=data_transform)
#testset = dsets.MNIST(root='.', train=False, download=True, transform=data_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)



n_gpu = args.GPU
print(n_gpu)
model = Net(7)
if n_gpu > 0:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
	if n_gpu > 0:
	    data, target = Variable(data.cuda()), Variable(target.cuda())
	elif n_gpu==0:
	    data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
	if n_gpu > 0:
            data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
	elif n_gpu==0:
	    data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    start_t = time.time()
    train(epoch)
    end_t = time.time()
    print('Total Train Time: {:.2f}sec'.format(end_t-start_t))
    validation()
    model_file = 'model/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')

