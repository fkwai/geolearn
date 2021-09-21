import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter as P
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.model import waterNet
import importlib

importlib.reload(waterNet)
P = torch.tensor([1, 2, 3])[:, None]
Q = torch.tensor([1, 2, 3])[:, None]/10
T = F.relu(torch.rand(3))[:, None]

wi = torch.rand(5)
LN = waterNet.LinearReservoir(5)
# x = F.relu(torch.rand(3, 5))
h = F.relu(torch.rand(3, 5))
s = F.relu(torch.rand(3, 5))
wm = torch.rand(5)
# if T > 0:
#     s = s+P
#     m = 0
# else:
#     m = T*F.sigmoid(wm)
#     s = s-m
m = torch.minimum(F.relu(T)*F.sigmoid(wm), s)
s = s-m+(T > 0)*P

x = P + m

qLst = list()
hLst = list()

P2 = torch.tensor([0, 0, 0])[:, None]
x = P2*F.sigmoid(wi)


for k in range(10):
    q, h = LN(x, h)
    qLst.append(torch.sum(q).detach().cpu().numpy())

fig, ax = plt.subplots(1, 1)
ax.plot(qLst)
fig.show()
#    if temp[k] < 0:
#         snow[k] = snow[k-1] + prcp[k]
#         liq[k] = 0
#     else:
#         snow[k] = max(snow[k-1]-d*temp[k], 0)
#         liq[k] = prcp[k] + min(snow[k], d*temp[k], 0.)
