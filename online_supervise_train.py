import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from online_supervise_model import haptac_online
from online_supervise_dataloader import sample_data
import pickle
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='../recordings/exp5/', help='Experiment path')
parser.add_argument('--exp_name', type=str, default='_01', help='Experiment name')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--window', type=int, default=200, help='window around the time step')
parser.add_argument('--window_z', type=int, default=800, help='window around the time step')
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--ckpt', type=str, default='val_online_best', help='loaded ckpt file')
parser.add_argument('--eval', type=bool, default=True, help='Set true if eval time')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
args = parser.parse_args()

if not os.path.exists(args.exp_dir + 'ckpts'):
    os.makedirs(args.exp_dir + 'ckpts')

if not os.path.exists(args.exp_dir + 'predictions'):
    os.makedirs(args.exp_dir + 'predictions')

if not args.eval:
    train_path = args.exp_dir + 'train_online' + args.exp_name + '.p'
    mask = []
    train_dataset = sample_data(train_path, args.window, args.window_z)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print(len(train_dataset))

    val_path = args.exp_dir + 'val_online' + args.exp_name + '.p'
    val_dataset = sample_data(val_path, args.window, args.window_z)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print(len(val_dataset))

if args.eval:
    test_path = args.exp_dir + 'test_online' + args.exp_name + '.p'
    test_dataset = sample_data(test_path, args.window, args.window_z)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print(len(test_dataset))

print(args.exp_dir, args.window)

'''training code'''
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda:0'
    model = haptac_online(4000, 1000, 1000, 1000)  # model
    model.to(device)
    best_train_loss = np.inf
    best_val_loss = np.inf

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()

    if args.train_continue:
        checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.ckpt + args.exp_name + '.path.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("ckpt loaded", loss, "Now continue training")

    '''evaluation and optimize'''
    if args.eval:
        eval_loss = []
        act_list = np.zeros((1, 5, 200))
        tactile_list = np.zeros((1, 5, 200))
        tactile_pred_list = np.zeros((1, 5, 200))
        checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.ckpt + args.exp_name + '.path.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("ckpt loaded:", args.ckpt, loss, "Now running on eval set")
        model.eval()

        bar = ProgressBar(max_value=len(test_dataloader))
        for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
            act_z = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
            tactile_z = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
            tactile_goal_z = torch.tensor(sample_batched[2], dtype=torch.float, device=device)
            act = torch.tensor(sample_batched[3], dtype=torch.float, device=device)
            tactile = torch.tensor(sample_batched[4], dtype=torch.float, device=device)

            with torch.set_grad_enabled(False):
                # tactile_pred, _ = model(act_z, tactile_z, tactile_goal_z, act)
                tactile_pred, _ = model(act_z, tactile_z, act)

            act_list = np.concatenate((act_list, act.cpu().data.numpy()), axis=0)
            tactile_list = np.concatenate((tactile_list, tactile.cpu().data.numpy()), axis=0)
            tactile_pred_list = np.concatenate((tactile_pred_list, tactile_pred.cpu().data.numpy()), axis=0)

            loss = criterion(tactile_pred, tactile) * 100

            eval_loss.append(loss.data.item())
            # pickle.dump([act_list, tactile_list, tactile_pred_list],
            #             open(args.exp_dir + 'predictions/eval_online' + args.exp_name + '/' + str(i_batch) + '.p',
            #                  "wb"))

        pickle.dump([act_list, tactile_list, tactile_pred_list],
                    open(args.exp_dir + 'predictions/eval_online' + args.exp_name +'.p', "wb"))

        print ('loss:', np.mean(eval_loss))

    else:
        writer = SummaryWriter(comment='online' + args.exp_name)
        n = 0

        for epoch in range(args.epoch):
            train_loss = []
            val_loss = []

            bar = ProgressBar(max_value=len(train_dataloader))

            for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
                model.train(True)
                act_z = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
                tactile_z = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
                tactile_goal_z = torch.tensor(sample_batched[2], dtype=torch.float, device=device)
                act = torch.tensor(sample_batched[3], dtype=torch.float, device=device)
                tactile = torch.tensor(sample_batched[4], dtype=torch.float, device=device)

                # print(act.size(), tactile.size())

                with torch.set_grad_enabled(True):
                    # tactile_pred, _ = model(act_z, tactile_z, tactile_goal_z, act)
                    tactile_pred, _ = model(act_z, tactile_z, act)

                # print(criterion(tactile_z, tactile_goal_z) * 100)
                loss = criterion(tactile_pred, tactile) * 100
                # print (loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.data.item())

                writer.add_scalar('Loss/train_perBatch', loss, epoch * len(train_dataloader) + i_batch)
                writer.add_scalar('Loss/train_meanSoFar', np.mean(train_loss),
                                  epoch * len(train_dataloader) + i_batch)
                # pickle.dump([act.cpu().data.numpy(), tactile.cpu().data.numpy(), tactile_pred.cpu().data.numpy()],
                #             open(args.exp_dir + 'predictions/train_' + str(args.window) + '_' + str(epoch) + '.p', "wb"))

                if i_batch % 20 == 0 and i_batch != 0:
                    val_loss_t = []
                    n += 1

                    print("[%d/%d], Loss: %.6f" % (
                        i_batch, len(train_dataloader),loss.item()))

                    # torch.save({
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'loss': loss, },
                    #     args.exp_dir + 'ckpts/train_online_' + str(args.window) + '_' + str(epoch) + args.exp_name + '.path.tar')

                    print("Now running on val set")
                    model.train(False)

                    bar = ProgressBar(max_value=len(val_dataloader))
                    for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):
                        act_z = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
                        tactile_z = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
                        tactile_goal_z = torch.tensor(sample_batched[2], dtype=torch.float, device=device)
                        act = torch.tensor(sample_batched[3], dtype=torch.float, device=device)
                        tactile = torch.tensor(sample_batched[4], dtype=torch.float, device=device)

                        with torch.set_grad_enabled(False):
                            # tactile_pred, _ = model(act_z, tactile_z, tactile_goal_z, act)
                            tactile_pred, _ = model(act_z, tactile_z, act)


                        loss = criterion(tactile_pred, tactile) * 100

                        if i_batch % 100 == 0 and i_batch != 0:
                            print("[%d/%d], Loss: %.6f, min: %.6f, mean: %.6f, max: %.6f"% (
                                i_batch, len(train_dataloader), loss.item(), np.amin(tactile_pred.cpu().data.numpy()),
                                np.mean(tactile_pred.cpu().data.numpy()), np.amax(tactile_pred.cpu().data.numpy())))

                        val_loss.append(loss.data.item())
                        val_loss_t.append(loss.data.item())
                    writer.add_scalar('Loss/val', np.mean(val_loss_t), n)

                    # scheduler.step(np.mean(val_loss))
                    if np.mean(train_loss) < best_train_loss:
                        print("new best train loss:", np.mean(train_loss))
                        best_train_loss = np.mean(train_loss)

                        # pickle.dump([act.cpu().data.numpy(), tactile.cpu().data.numpy(), tactile_pred.cpu().data.numpy()],
                        #             open(args.exp_dir + 'predictions/train_online_' + str(args.window) + '_' + str(epoch) + args.exp_name + '.p', "wb"))
                        pickle.dump([act.cpu().data.numpy(), tactile.cpu().data.numpy(), tactile_pred.cpu().data.numpy()],
                                    open(args.exp_dir + 'predictions/train_online_best' + args.exp_name + '.p', "wb"))
                        # print (tactile, tactile_pred)

                    # print ("val_loss:", np.mean(val_loss))
                    if np.mean(val_loss_t) < best_val_loss:
                        print("new best val loss:", np.mean(val_loss_t))
                        best_val_loss = np.mean(val_loss_t)

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss, },
                            args.exp_dir + 'ckpts/val_online_best' + args.exp_name + '.path.tar')

                avg_train_loss = np.mean(train_loss)
                avg_val_loss = np.mean(val_loss)

                # avg_train_loss = np.array([avg_train_loss])
                # avg_val_loss = np.array([avg_val_loss])
                #
                # train_loss_list = np.append(train_loss_list, avg_train_loss, axis=0)
                # val_loss_list = np.append(val_loss_list, avg_val_loss, axis=0)

            print("Train Loss: %.6f, Valid Loss: %.6f" % (avg_train_loss, avg_val_loss))
