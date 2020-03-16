import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit
import matplotlib.pyplot as plt

caseName = 'refBasins'
# load data
dictData, info, x, y, c = waterQuality.loadData(caseName)

varX = ['streamflow', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
varY = dictData['varC']
varC = dictData['varG']

for k, var in enumerate(varX):
    fig, axes = plt.subplots(2, 1)
    temp = x[:, :, k].flatten()
    axes[0].hist(temp, bins=200)
    axes[1].hist(np.log(temp+1), bins=200)
    fig.suptitle(var)
    fig.show()
# q and p, log-stan; others stan

for k, var in enumerate(varY):
    fig, axes = plt.subplots(2, 1)
    temp = y[:, k].flatten()
    if not np.isnan(temp).all():
        if var == '00010':
            temp = temp+40
        axes[0].hist(temp, bins=200)
        axes[1].hist(np.log(temp+1), bins=200)
    fig.suptitle(str(k)+' '+var)
    fig.show()

temp = y[:, 11].flatten()
temp=temp[temp<10000]
fig, axes = plt.subplots(2, 1)
axes[0].hist(temp, bins=200)
axes[1].hist(np.log(temp+1), bins=200)
fig.show()


for k, var in enumerate(varC):
    fig, axes = plt.subplots(2, 1)
    temp = c[:, k].flatten()
    if not np.isnan(temp).all():
        if var == '00010':
            temp = temp+40
        axes[0].hist(temp, bins=200)
        axes[1].hist(np.log(temp+1), bins=200)
    fig.suptitle(str(k)+' '+var)
    fig.show()