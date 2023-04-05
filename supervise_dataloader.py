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

        # act = np.reshape(act, (act.shape[0], act.shape[1], 1))
        # tactile = np.reshape(tactile, (tactile.shape[0], tactile.shape[1], 1))
        # act = np.reshape(act, (1, act.shape[0], act.shape[1]))
        # tactile = np.reshape(tactile, (1, tactile.shape[0], tactile.shape[1]))

        return act, tactile