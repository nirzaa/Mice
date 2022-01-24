import torch.nn as nn
import torch.nn.functional as F
import torch

class MiceConv(nn.Module):
    def __init__(self, input_size=576):
        super().__init__()

        self.layer1 =  nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
                                     nn.ReLU())
        

        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.fc2 = nn.Linear( int(input_size/2),  1)
        
        print('Finished init')

    def forward(self, data):
        output = self.layer1(data)
        output = self.layer2(output)

        output = self.drop_out(output)
        output = output.reshape(output.size(0), -1)

        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

class Net(nn.Module):
    '''
    The linear architecture of the neural net
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, 2 * 2 * 2)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x


class Model(nn.Module):
    '''
    The semi fully conventional architecture of the neural net
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(2,2), stride=(1,1), padding=1,)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1, )
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1, )
        self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global1(x)
        x = self.avgpool(x)
        x = x.view(-1, 2 * 2 * 2)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x


class Modely(nn.Module):
    '''
    The real fully conventional architecture of the neural net
    '''
    def __init__(self):
        super(Modely, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=1,)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding=1, )
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=1, )
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=(1,1,1), stride=(1,1,1), padding=0, )
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, )
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout1 = nn.Dropout(p=0.2) # p – probability of an element to be zeroed. Default: 0.5
        self.batchnorm1 = nn.BatchNorm3d(num_features=16)
        self.batchnorm2 = nn.BatchNorm3d(num_features=32)
        self.batchnorm3 = nn.BatchNorm3d(num_features=64)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        # torch.Size([32, 64, 2, 2,2])
        x = self.avgpool3d(x)
        # torch.Size([32, 64, 1, 1, 1])
        x = self.conv4(x)
        x = self.conv5(x)
        # torch.Size([32, 1, 1, 1])
        x = x.view(-1, 1)
        # torch.Size([32, 1])
        return x
