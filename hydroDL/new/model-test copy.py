import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import exp, lgamma
from hydroDL.new.model import flowPath
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
