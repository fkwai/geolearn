import math
import torch
# import torch.nn as nn
from torch.nn.parameter import Parameter as P
import torch.nn.functional as F


class RnnCell(torch.nn.Module):
    def __init__(self, *, nh):
        super().__init__()
        self.hiddenSize = nh
        self.w_i = P(torch.Tensor(nh, 1))
        self.w_h = P(torch.Tensor(nh, nh))
        self.w_o = P(torch.Tensor(1, nh))
        self.b_i = P(torch.Tensor(nh))
        self.b_h = P(torch.Tensor(nh))
        self.b_o = P(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            # w.data.uniform_(-1, 1)

    def forward(self, x, h):
        h1 = F.linear(x, self.w_i, self.b_i) + F.linear(h, self.w_h, self.b_h)
        y1 = F.linear(h1, self.w_o, self.b_o)
        return y1, h1


class RnnModel(torch.nn.Module):
    def __init__(self, nh):
        super(RnnModel, self).__init__()
        # self.rnn = WaterNetCell(nh=nh)
        self.rnn = RnnCell(nh=nh)
        self.nh = nh
        self.h0 = P(torch.Tensor(nh, 1))

    def forward(self, x):
        nt, nb = x.shape
        h0 = torch.zeros(nb, self.nh).cuda()
        y = torch.zeros(nt, nb).cuda()
        h = torch.zeros(nt, nb, self.nh).cuda()
        for k in range(nt):
            y1, h1 = self.rnn(x[k, :, None], h0)
            y[k, :] = y1[:, 0]
            h[k, :, :] = h1
            h0 = h1
        return y, h


class WaterNetCell(torch.nn.Module):
    def __init__(self, *, nh):
        super().__init__()
        self.hiddenSize = nh
        self.w_i = P(torch.Tensor(nh, 1))
        self.w_o = P(torch.Tensor(1, nh))
        self.b_i = P(torch.Tensor(nh))
        self.b_o = P(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        # std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            # w.data.uniform_(-std, std)
            w.data.uniform_(0, 1)

    def forward(self, x, h):
        x1 = F.relu(x*self.w_i.T+self.b_i)
        h1 = x1+h
        q = F.relu(h1*self.w_o)
        h1 = F.relu(h1-q)
        # print(h1)
        y1 = torch.mean(q, dim=-1, keepdim=True) + self.b_o
        return y1, h1


class LinearReservoir(torch.nn.Module):
    def __init__(self, nh, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.w = P(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.uniform_(-1, 1)

    def forward(self, x, h):
        q = (x+h)*torch.sigmoid(self.w)
        hn = h-q+x
        return q, hn


class SnowReservoir(torch.nn.Module):
    def __init__(self, nh, nm=1, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.nm = nm
        self.w = P(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.uniform_(-1, 2)

    def forward(self, s, P, T):
        sm = F.relu(T[:, None]) * torch.pow(self.w, 10)
        m1 = sm[:, :self.nm]
        m2 = torch.minimum(sm[:, self.nm:], s[:, self.nm:])
        m = torch.cat((m1, m2), dim=1)
        s = s-m+((T < 0)*P)[:, None]
        x = ((T > 0)*P)[:, None] + m
        return x, s


class WaterNetModel(torch.nn.Module):
    def __init__(self, nh, *, nm=0):
        super().__init__()
        self.nh = nh
        self.w_i = P(torch.Tensor(nh))
        self.w_o = P(torch.Tensor(nh))
        self.w_e = P(torch.Tensor(nh))
        self.LN = LinearReservoir(nh)
        self.SN = SnowReservoir(nh, nm)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_i.data.uniform_(-1, 1)
        self.w_o.data.uniform_(-1, 1)
        self.w_e.data.uniform_(-1, 1)

    def forward(self, P, T, Q0=0):
        # initial states
        nt, ns = P.shape
        S = torch.zeros(nt, ns, self.nh).cuda()
        H = torch.zeros(nt, ns, self.nh).cuda()
        Q = torch.zeros(nt, ns).cuda()
        s = torch.zeros(ns, self.nh).cuda()
        h = torch.zeros(ns, self.nh).cuda()
        for k in range(nt):
            x, sn = self.SN(s, P[k, :], T[k, :])
            S[k, :, :] = sn
            xin = x*torch.sigmoid(self.w_i)
            q, hn = self.LN(xin, h)
            H[k, :, :] = hn
            Q[k, :] = torch.sum(q*F.softmax(self.w_o, dim=0), dim=1)
            s = sn
            h = hn
        return Q, H, S
