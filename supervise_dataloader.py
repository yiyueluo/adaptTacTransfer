'''
This code defines the dataloder for the forward model (without adaptation module),
which will be used in supervise_train.py, along with supervise_model.py.
'''


import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle


def window_select(data,timestep,window):
    if window == 0:
        return data[:, timestep : timestep + 1]
    max_len = data.shape[1]
    u = min(max_len,timestep+window)
    if u == max_len:
        return (data[:, -window:])
    else:
        return(data[:, timestep:u])


class sample_data(Dataset):
    def __init__(self, path, window):
        self.path = path
        self.data = pickle.load(open(self.path, "rb"))
        self.window = window

    def __len__(self):
        # return self.length
        return self.data[0].shape[1]
        # return 1

    def __getitem__(self, idx):
        act = window_select(self.data[3],idx,self.window) #0 3
        tactile = window_select(self.data[4],idx,self.window) #1 4

        return act, tactile