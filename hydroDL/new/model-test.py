import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import exp, lgamma
from hydroDL.new.model import flowPath
import importlib
importlib.reload(flowPath)

nt = 1000
rho = 365
nb = 100
nx = 1
nq = 40

x = torch.randn(nt, nb, nx).cuda()
y = torch.random(nt, nb, 1).cuda()

hiddenSize = 64
inputSize = nx
convSize = nq
batchSize = x.shape[1]

model = flowPath(nx, 256, nq).cuda()
model(x, rho)

# rnn = nn.RNN(inputSize, hiddenSize)
# linear = torch.nn.Linear(hiddenSize, convSize)
# aT = exp(Parameter(torch.Tensor(nq)))
# bT = exp(Parameter(torch.Tensor(nq)))

# out1, hn = rnn(x)
# out2 = linear(out1)
# xT = torch.arange(1, rho+1, dtype=torch.float32)/(rho+1)
# x1 = exp(lgamma(aT+bT)-lgamma(aT)-lgamma(bT)).view(-1, 1).expand(-1, rho)
# x2 = xT.view(1, -1).expand(nq, -1)**(aT.view(-1, 1).expand(-1, rho)-1)
# x3 = (1-xT.view(1, -1).expand(nq, -1))**(bT.view(-1, 1).expand(-1, rho)-1)
# qT = x1*x2*x3
# out = F.conv1d(out2.permute(1, 2, 0), qT[None, :, :])

# w_xh = Parameter(torch.Tensor(hiddenSize, inputSize))
# w_hh = Parameter(torch.Tensor(hiddenSize, hiddenSize))
# w_ho = Parameter(torch.Tensor(hiddenSize, 1))
# b_h = Parameter(torch.Tensor(hiddenSize))
# b_o = Parameter(torch.Tensor(1))

# if h0 is None:
#     h0 = x.new_zeros(batchSize, hiddenSize, requires_grad=False)
# h = F.linear(x0, w_xh) + + F.linear(h0, w_hh) + b_h
