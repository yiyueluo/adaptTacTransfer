'''
This code runs the invers optimization for haptic output (with or without adaptation module),
which outputs haptic signal based on defined target tactile signal and trained forward model.
'''

import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from online_supervise_model import haptac_online
import pickle
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from supervise_model import haptac
from OnlineSenHaptExtract import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--adapt', type=bool, default=True, help='Set true if would like to have the adaptation module')
parser.add_argument('--exp_dir', type=str, default='./exp/', help='Experiment path')
parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--window', type=int, default=200, help='Window around the time step')
parser.add_argument('--n_iter', type=int, default=5000, help='Set number of iteractions')
parser.add_argument('--ckpt', type=str, default='val_best_adapt_sample', help='Loaded forward model ckpt file')
args = parser.parse_args()


def cal(p):
    base = np.median(p[:150, :, :], axis=0)
    p = p - base
    p = (p - np.quantile(p, 0.1, axis=0)) / (np.amax(p, axis=0) - np.quantile(p, 0.1, axis=0))
    return p

def act_transform(act):
    n_finger = 5
    t = act.shape[0]
    output = np.zeros((n_finger, t))
    for i in range(t):
        if act[i] >= 0:
            output[np.int16(act[i]), i] = 1

    return output

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def sample_hapt(tac_goal, space):
    hapt_goal = []
    for i in range(tac_goal.shape[1]):
        ind = np.argmax(tac_goal[:, i])
        hapt_goal.append(ind)
    output = []
    output.append(hapt_goal)
    for i in range(1, space):
        s = []
        s =  hapt_goal[:-i] + [0 for iter in range(i)]
        # print (len(s))
        output.append(s)

    return output


def sample_hapt_multifinger(tac_goal, space):
    hapt_goal = []
    for i in range(tac_goal.shape[1]):
        sub = [-1]
        for j in range(tac_goal.shape[0]):
            if tac_goal[j, i] > 0.3:
                sub.append(j)
        hapt_goal.append(sub)

    output = []
    output.append(hapt_goal)
    intermediate = [-1]
    for i in range(1, space):
        s =  hapt_goal[:-i] + [intermediate for iter in range(i)]
        output.append(s)

    return output

def hapt_transform(sample_act):
    batch = len(sample_act)
    n_finger = 5
    t = len(sample_act[0])
    output = np.zeros((batch, n_finger, t))

    for b in range(batch):
        for i in range(t):
            output[b, np.int16(sample_act[b][i]), i] = 1

    return output

def hapt_transform_multifinger(sample_act):
    batch = len(sample_act)
    n_finger = 5
    t = len(sample_act[0])
    output = np.zeros((batch, n_finger, t))

    for b in range(batch):
        for i in range(t):
            for k in range(len(sample_act[b][i])):
                if sample_act[b][i][k] != -1:
                    output[b, np.int16(sample_act[b][i][k]), i] = 1

    return output

def smooth_torch(act):
    mask = nn.Conv2d(32, 32, kernel_size=(1,5),padding=(0, 2), dtype=torch.float, device='cuda:0')
    output = mask(act)
    output = torch.where(output> 0.5, 1, 0)

    return output


'''optimization code'''
if __name__ == '__main__':

    act_optimized_output = np.zeros((5, 50)) #optimized haptic sequence
    tactile_output = np.zeros((5, 50)) #tactile goal sequence
    tactile_pred_output = np.zeros((5, 50)) #predicted tactile sequence by forward model
    act_unoptimized_output = np.zeros((5, 50)) #unoptimzied haptic sequence from tactile goal
    act_ori_output = np.zeros((5, 50)) #haptic input extracted from tactile goal

    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda:0'
    if args.adapt:
        model = haptac_online(4000, 1000, 1000, 1000)  # online model
    else:
        model = haptac(1000, 1000)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()

    checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.ckpt + args.exp_name + '.path.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    loss = checkpoint['loss']
    print("ckpt loaded:", args.ckpt, loss, "Now running on val set")
    model.eval()

    bar = ProgressBar(max_value=args.n_iter)

    # tac_goal = generate_states(length=4, num_finger=5, repeat_bool=False, space_bool=False)
    # repeat = 40
    space = 32
    # tac_goal = [element for element in a for i in range(repeat)]
    # tac_goal = tac_goal[space:]

    # data = pickle.load(open(args.exp_dir + 'test.p', "rb"))
    data = pickle.load(open(args.exp_dir + 'test_online' + args.exp_name + '.p', "rb"))
    # data = pickle.load(open(args.exp_dir + 'op_online' + args.exp_name + '.p', "rb"))
    # print (data[0].shape)

    # st = 200
    # ed = 200-50t
    st = 0
    ed = data[0].shape[1]
    for i in range(st, ed-200, 200):
        m = min(i+800, ed)
        act_z = data[0][:, m-800:m]
        tactile_z = data[1][:, m-800:m]
        tactile_goal_z = data[2][:, m-800:m]

        act_ori = data[3][:, i:i+200]
        tac_goal = data[4][:, i:i+200]
        sample_act = sample_hapt_multifinger(tac_goal, space)
        act = hapt_transform_multifinger(sample_act)
        act_unoptimized = act[0, :, :]

        # print (tac_goal.shape, tactile_goal_z.shape)

        # for i in range(tac_goal.shape[0]):
        #     plt.plot(tac_goal[i, :])
        # plt.plot(sample_act[0])
        # plt.show()

        # for i in range(act.shape[1]):
        #     plt.plot(act[0, i, :])
        # plt.show()

        act = torch.tensor(act, dtype=torch.float, requires_grad=True, device=device)
        act_ori = torch.tensor(np.tile(act_ori, (32, 1, 1)), dtype=torch.float, device=device)
        tactile = torch.tensor(np.tile(tac_goal, (32, 1, 1)), dtype=torch.float, device=device)
        act_z = torch.tensor(np.tile(act_z, (32, 1, 1)), dtype=torch.float, device=device)
        tactile_z = torch.tensor(np.tile(tactile_z, (32, 1, 1)), dtype=torch.float, device=device)
        tactile_goal_z = torch.tensor(np.tile(tactile_goal_z, (32, 1, 1)), dtype=torch.float, device=device)


        optimizer = optim.Adam([act], lr=1e-3, betas=(0.9, 0.999))

        for idx_iter in range(args.n_iter):
            with torch.set_grad_enabled(True):
                if args.adapt:
                    tactile_pred, _ = model(act_z, tactile_z, tactile_goal_z, act)
                    loss = criterion(tactile_pred, tactile) * 100 + torch.std(tactile_pred)
                else:
                    tactile_pred = model(act)
                    loss = criterion(tactile_pred, tactile) * 100 + torch.std(tactile_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                act.data.clamp_(min=0, max=1)
                # act.data.to(dtype=torch.float)
                # act.data = smooth_torch(act.data)


        print(loss)

        data_output = [act.cpu().data.numpy(), tactile.cpu().data.numpy(), tactile_pred.cpu().data.numpy(),
                     act_ori.cpu().data.numpy(), act_unoptimized]

        act_optimized_output = np.concatenate((act_optimized_output[:, :-50], np.mean(data_output[0], axis=0)), axis=1)
        tactile_output = np.concatenate((tactile_output[:, :-50], np.mean(data_output[1], axis=0)), axis=1)
        tactile_pred_output = np.concatenate((tactile_pred_output[:, :-50], np.mean(data_output[2], axis=0)), axis=1)
        act_ori_output = np.concatenate((act_ori_output[:, :-50], np.mean(data_output[3], axis=0)), axis=1)
        act_unoptimized_output = np.concatenate((act_unoptimized_output[:, :-50], data_output[4]), axis=1)


        pickle.dump([act_optimized_output, tactile_output, tactile_pred_output, act_ori_output, act_unoptimized_output],
                    open(args.exp_dir + 'optimize/optimized' + args.exp_name + '.p', "wb"))









