import torch
import torchvision
from torch import nn, optim
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        #self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8*12*12, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #x = self.conv3(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        x = x.view(-1, 8*12*12)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x 
    

if __name__== "__main__":
    device = torch.device("cpu")
    model = CNN(2).to(device)
    summary(model, (3, 54, 54))