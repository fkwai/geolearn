import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin, gageII, usgs, gridMET
import torch
import importlib
from hydroDL.model import waterNet, crit

# test case
siteNo = '07241550'
siteNo = '10343500'
varLst = gridMET.varLst+usgs.varC+usgs.varQ
df = dbBasin.readSiteTS(siteNo, varLst)
P = df['pr'].values
Q = df['runoff'].values*1000/365
E = df['etr'].values
np.mean(P-E)

nt = len(P)
nh = 3
fMat = np.array([1, 0.5, 1])
kMat = np.array([0.1, 0.5, 0.5])
R = np.array([0.5, 0.1, 0.3])

# fMat = np.random.rand(3)
# kMat = np.random.rand(3)
# # R = 1/kMat
# R = np.random.rand(3)

H = np.ones([nt, nh])
Y = np.zeros(nt)
C = np.zeros(nt)

H[0, :] = Q[0]/kMat
for k in range(1, nt):
    H[k, :] = H[k-1, :] + P[k]*fMat
    q = H[k, :]*kMat
    Y[k] = np.mean(q)
    H[k, :] = H[k, :]-q
    C[k] = np.mean(q*R)/np.mean(q)

# fig, axes = plt.subplots(4, 1)
# axes[0].plot(P)
# axes[1].plot(Q)
# axes[2].plot(Y)
# axes[3].plot(C)
# fig.show()


fig, axes = plt.subplots(2, 1)
axes[0].plot(Y, C, '*')
axes[1].plot(np.log(Y), C, '*')
fig.show()
