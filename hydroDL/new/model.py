import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import exp, lgamma


class flowPath(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, convSize):
        super(flowPath, self).__init__()
        self.rnn = nn.RNN(inputSize, hiddenSize)
        # self.rnn = nn.LSTM(inputSize, hiddenSize)
        self.linear = torch.nn.Linear(hiddenSize, convSize)
        a0 = torch.Tensor([1, 1, 1])
        b0 = torch.Tensor([.1, 1, 10])
        self.aT = Parameter(a0.cuda())
        self.bT = Parameter(b0.cuda())
        self.convSize = convSize
        # self.reset_parameters()

    def forward(self, x, rho):
        out1, hn = self.rnn(x)
        # out2 = F.relu(self.linear(out1))
        out2 = self.linear(out1)
        xT = (torch.arange(1, rho+1, dtype=torch.float32)/(rho+1)).cuda()
        aT = exp(self.aT)
        bT = exp(self.bT)
        # aT = self.aT
        # bT = self.bT
        nq = self.convSize
        x1 = exp(lgamma(aT+bT)-lgamma(aT)-lgamma(bT)
                 ).view(-1, 1).expand(-1, rho)
        x2 = xT.view(1, -1).expand(nq, -1)**(aT.view(-1, 1).expand(-1, rho)-1)
        x3 = (1-xT.view(1, -1).expand(nq, -1)
              )**(bT.view(-1, 1).expand(-1, rho)-1)
        qT = x1*x2*x3
        out = F.conv1d(out2.permute(1, 2, 0), qT[None, :, :])
        return out.permute(2, 0, 1)
