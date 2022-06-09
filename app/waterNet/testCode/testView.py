import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
nt = 4
nh = 3
ns = 2
nc = 5
q = torch.randn(nt, ns, nh)
c = torch.randn(ns, nh, nc)

q.view(nt, ns*nh)*c.view(ns*nh, nc)

# numpy
qn = q.numpy()
cn = c.numpy()
yn = np.zeros([nt, ns, nh, nc])
for iT in range(nt):
    qq = qn[iT, :, :]
    for iC in range(nc):
        cc = cn[:, :, iC]
        yn[iT, :, :, iC] = qq*cc


a = q[:, :, :, None]
b = c.repeat(nt, 1, 1, 1)

y = a*b
y.numpy()-yn


fcC = nn.Sequential(
    nn.Linear(ng, 256),
    nn.Tanh(),
    nn.Linear(256, nm*nc*3)).cuda()

torch.rand(20, 10)
