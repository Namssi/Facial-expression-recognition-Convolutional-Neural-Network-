import torch
import torch.nn as nn
import torch.nn.functional as F




class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        #self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*4*4,64* 4*4)
        self.fc2 = nn.Linear(64*4*4, output_size)

        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.fc1_bn = nn.BatchNorm2d(64*4*4)


    def forward(self, x):
        #x = x.view(-1, self.input_size)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1_bn(x)

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv2_bn(x)

        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.conv3_bn(x)

        x = x.view(-1, 256*4*4)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        #x = F.dropout(x, training=self.training)

        return F.sigmoid(self.fc2(x))


