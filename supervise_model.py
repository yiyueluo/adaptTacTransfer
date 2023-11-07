'''
This code defines the forward model (without adaptation module),
which will be used in supervise_train.py with supervise_dataloader.py..
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class haptac(nn.Module):
    def __init__(self, num_input, num_output):
        super(haptac, self).__init__()
        self.linear0 = nn.Linear(num_input,1280) #6x3=18
        self.linear1 = nn.Linear(1280,2560)
        self.linear2 = nn.Linear(2560,1280)
        self.linear3 = nn.Linear(1280,num_output) #6x15=90

    def forward(self, states):
        """Forward pass."""
        x = states.reshape(states.shape[0], -1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = x.reshape(states.shape[0], states.shape[1], states.shape[2])

        return x