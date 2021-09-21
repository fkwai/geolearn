import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
import torch
import importlib
from hydroDL.model import waterNet, crit

# test case
siteNo = '07241550'
df = dbBasin.readSiteTS(siteNo, ['pr', 'runoff', 'etr'])
P = df['pr'].values
Q = df['runoff'].values*1000/365
E = df['etr'].values
np.mean(P-E)

nt = len(P)
nh = 3
kMat = np.array([0.1, 0.5, 0.9])/10
H = Q[:, None]/kMat[None, :]

fig, axes = plt.subplots(4, 1)
axes[0].plot(Q)
axes[1].plot(H[:, 0])
axes[2].plot(H[:, 1])
axes[3].plot(H[:, 2])
fig.show()
