'''
This code visualize the optimization results.

It visualize the target tactile signal in solid lines, unoptimized haptic signal in dashed lines (sqaure waves),
and optimized haptic signal in solid lines (square waves).

'''

import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

c = ["#3951a2", "#5c91c2", "#fdb96b", '#f67948', "#da382a"]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def cal(p):
    base = np.mean(p[:200, :, :], axis=0)
    p = p - base
    p = np.where(p>300, 0, p)
    p = (p - np.quantile(p, 0.1, axis=0)) / (np.amax(p, axis=0) - np.quantile(p, 0.1, axis=0))
    p = np.clip(p, 0, 1)
    return p


''' viz optimize result'''
# define the optimized output here
name = '_sample'
# path = '../recordings/exp5/optimize/online_best' + name + '.p'
path = '../recordings/exp5/optimize/optimized' + name + '.p'


data = pickle.load(open(path, "rb"))
act = data[0]
tactile = data[1]
tactile_pred = data[2]
act_ori = data[3]
act_unoptimized = data[4]

print (act.shape, tactile.shape, tactile_pred.shape, act_unoptimized.shape)

for j in range(act.shape[0]):
    act[j, :] = smooth(act[j, :], 40)
act = np.where(act>0.6, 1, 0)

baseline = np.copy(act)
baseline[:] = 0
mse = mean_squared_error(act, act_ori)
mse_baseline = mean_squared_error(baseline, act_ori)
print (mse, mse_baseline)

for i in range(5):
    mse = mean_squared_error(act[i, :], act_ori[i, :])
    print (mse)

fig, ax = plt.subplots(figsize=(10, 10))
fig.tight_layout(pad=1)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

for i in range(5):
    ax.plot(act_unoptimized[i, :] + i * 2.5, color=c[i], linestyle='dashed')
    ax.plot(act[i, :] + i * 2.5, color=c[i])
    ax.plot(smooth(tactile_pred[i, :], 20) + i * 2.5, color=c[i])
plt.show()
