from hydroDL.post import axplot, figplot
from hydroDL.new import fun
from hydroDL.app import waterQuality
import importlib
import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np
import random
import torch
import torch.nn.functional as F


# temp
p = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [3, 4, 5, 6]])
qc = np.array([[0, 1], [2, 5], [4, 3]])
qc2 = torch.from_numpy(qc[None, :, :]).float()
p2 = torch.from_numpy(p[None, :]).float()

F.conv1d(p2, qc2).numpy()
np.convolve(np.flip(qc[0, :]), p[0, :], 'valid')
np.convolve(np.flip(qc[1, :]), p[1, :], 'valid')
np.convolve(np.flip(qc[2, :]), p[2, :], 'valid')

np.convolve([4, 3], [1, 1, 5, 5], 'valid')


# random
nq = 3
nt = 100
rho = 10

p = np.random.random([nq, nt])
qc = np.random.random([nq, rho])
qc2 = torch.from_numpy(qc[None, :, :]).float()
p2 = torch.from_numpy(p[None, :]).float()
out2 = F.conv1d(p2, qc2).numpy()[0, 0, :]
outLst = list()
for k in range(nq):
    temp = np.convolve(np.flip(qc[k, :]), p[k, :], 'valid')
    outLst.append(temp)
outMat = np.stack(outLst, axis=-1)
out = np.sum(outMat, axis=1)
out-out2
