import torch
import torch.nn as nn
import torch.nn.functional as F

class haptac(nn.Module):
    """Dense neural network class."""
    def __init__(self, num_input, num_output):
        super(haptac, self).__init__()
        # self.conv0 = nn.Conv2d(num_input, 32, kernel_size=(3,3),padding=1)
        # self.conv1 = nn.Conv2d(32, 64, kernel_size=(3,3),padding=1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3),padding=1)
        # self.conv3 = nn.Conv2d(64, 32, kernel_size=(3,3),padding=1)
        # self.conv4 = nn.Conv2d(32, num_output, kernel_size=(3,3),padding=1)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

        self.linear0 = nn.Linear(num_input,1280) #6x3=18
        self.linear1 = nn.Linear(1280,2560)
        self.linear2 = nn.Linear(2560,1280)
        self.linear3 = nn.Linear(1280,num_output) #6x15=90

    def forward(self, states):
        """Forward pass."""
        # x = self.relu(self.conv0(x))
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = self.relu(self.conv4(x))

        x = states.reshape(states.shape[0], -1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = x.reshape(states.shape[0], states.shape[1], states.shape[2])

        return x