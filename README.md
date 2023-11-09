# Adaptive tactile interactions transfer

## Dependencies
Real-time demo needs to be run with our smart gloves and driving circuits. Off-line demos can be run with the provided sample data. 

Our code runs on Python3. Required packages include:

```
numpy
scipy
pytorch
opencv
pickle
h5py
argparse
```

## Installation Guide

Intall by runing
```git clone https://github.com/yiyueluo/adaptTacTransfer.git```.

Installation will take less than 2 minutes. 

## Data

Data can be downloaded from <a href="https://www.dropbox.com/sh/etfs0se726n0m0h/AABI6ENLPZ1Cv5LlAsFfNZRRa?dl=0" target="_blank">here</a>.

This includes sample data and trained checkpoint. 


## Training
Train a model with adaptation module by running ```./adapt_supervise_train.py```.

Train a model without adaptation module by running ```./supervise_train.py```.

Run inverse optimization by running ```./optimize.py```, where you can set ```--adapt``` to ```True``` if using adaptation module. 

## Visualization
Visualize the optimized haptic signal by running ```./optimize_viz.py```.

## Contact
Yiyue Luo: yiyueluo@mit.edu
