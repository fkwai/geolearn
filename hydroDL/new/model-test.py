import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import exp, lgamma
from hydroDL.new.model import flowPath
from hydroDL.model import crit
import importlib
importlib.reload(hydroDL.new.model)
rho = 1000
nd = 365
nb = 100
nx = 1
nq = 3

x = torch.randn(rho, nb, nx).cuda()
y = torch.randn(rho, nb, 1).cuda()

hiddenSize = 16
inputSize = nx
convSize = nq
batchSize = x.shape[1]
try:
    model = flowPath(nx, 256, nq).cuda()
except:
    pass
model = flowPath(nx, 256, nq).cuda()

type(model.aT)
type(model.rnn.weight_ih_l0)

lossFun = crit.RmseLoss().cuda()
optim = torch.optim.Adadelta(model.parameters())

# yp = model(x, nd)
# optim.zero_grad()
# loss = lossFun(yp, y[nd-1:, :, :])
# loss.backward()

for k in range(100):
    yp = model(x, nd)
    optim.zero_grad()
    loss = lossFun(yp, y[nd-1:, :, :])
    loss.backward()
    optim.step()
    print(loss.item())
    # model.aT
    model.bT

model.rnn.weight_ih_l0
model.aT
type(model.aT)

model.aT.grad
# model.rnn.weight_ih_l0.grad
