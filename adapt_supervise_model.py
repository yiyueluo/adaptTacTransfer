'''
This code defines the forward model (with adaptation module),
which will be used in online_supervise_train.py with oneline_supervise_dataloader.py.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class haptac_online(nn.Module):
    def __init__(self, num_input_z, num_input, num_z, num_output):
        super(haptac_online, self).__init__()
        self.linear0_z = nn.Linear(num_input_z * 2,1280) #6x3=18
        self.linear1_z = nn.Linear(1280,2560)
        self.linear2_z = nn.Linear(2560,1280)
        self.linear3_z = nn.Linear(1280,num_z) #6x15=90

        self.linear0 = nn.Linear(num_input + num_z, 1280) #6x3=18
        self.linear1 = nn.Linear(1280,2560)
        self.linear2 = nn.Linear(2560,1280)
        self.linear3 = nn.Linear(1280,num_output) #6x15=90

    def forward(self, act_z, tactile_z, act): #act_z, tactile_z, tactile_goal_z, act
        """Forward pass."""
        x = torch.cat((act_z, tactile_z), axis=2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.linear0_z(x))
        x = F.relu(self.linear1_z(x))
        x = F.relu(self.linear2_z(x))
        z = F.relu(self.linear3_z(x))

        x = act.reshape(act.shape[0], -1)
        x = torch.cat((x, z), axis=1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = x.reshape(act.shape[0], act.shape[1], act.shape[2])

        return x, z
