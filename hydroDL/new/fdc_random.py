from hydroDL.post import axplot, figplot
from hydroDL.new import fun
from hydroDL.app import waterQuality
import importlib
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from scipy.special import gamma, loggamma
import torch
import torch.nn.functional as F
from torch import exp, lgamma

# fake data
nq = 10
rho = 365
nt = 1000
nbatch = 30

p = np.random.random([nq, nt])
aAry = np.exp((np.random.random(nq)-0.5)*2)
bAry = np.exp((np.random.random(nq)-0.5)*2)

# numpy
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
qMat = np.ndarray([10, 365])
for k in range(10):
    a = aAry[k]
    b = bAry[k]
    x = (np.arange(365)+1)/366
    q = gamma(a+b)/gamma(a)/gamma(b)*x**(a-1)*(1-x)**(b-1)
    qMat[k, :] = q
    t = np.arange(365)
    ax.plot(t, q, label='a={:.3f} b={:.3f}'.format(a, b))
ax.legend()
fig.show()
outLst = list()
for k in range(10):
    temp = np.convolve(p[k, :], qMat[k, :].T, 'valid')-1
    outLst.append(temp)
outMat = np.stack(outLst, axis=0)

# torch
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
qMat = np.ndarray([10, 365])
pT = torch.tensor(p)
aT = torch.tensor(aAry)
bT = torch.tensor(bAry)
xT = torch.range(1, 365, dtype=torch.float64)/366
x1 = exp(lgamma(aT+bT)-lgamma(aT)-lgamma(bT)).view(-1, 1).expand(-1, 365)
x2 = xT.view(1, -1).expand(10, -1)**(aT.view(-1, 1).expand(-1, 365)-1)
x3 = (1-xT.view(1, -1).expand(10, -1))**(bT.view(-1, 1).expand(-1, 365)-1)
qT = x1*x2*x3
for k in range(10):
    t = xT.numpy()
    q = qT[k, :].numpy()
    ax.plot(t, q, label='a={:.3f} b={:.3f}'.format(a, b))
ax.legend()
fig.show()
outT=F.conv1d(pT[None, :, :], qT[None, :, :])


# # log - haven't figure out
# qMat2 = np.ndarray([10, 365])
# for k in range(10):
#     a = aAry[k]
#     b = bAry[k]
#     x = (np.arange(365)+1)/366
#     q2 = loggamma(a+b)+(a-1)*x+(b-1)*(1-x)-loggamma(a)-loggamma(b)
#     qMat2[k, :] = q2
# p2 = np.log(p)
# outLst2 = list()
# for k in range(10):
#     temp = np.convolve(p2[k, :], qMat2[k, :], 'valid')-1
#     outLst2.append(temp)
# outMat2 = np.stack(outLst2, axis=0)
