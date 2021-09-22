import math
import torch
import torch.nn as nn
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


class LinearBucket(torch.nn.Module):
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


class PowerBucket(torch.nn.Module):
    def __init__(self, nh, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.wk = P(torch.Tensor(nh))
        self.wa = P(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.wk.data.uniform_(-1, 1)
        self.wa.data.uniform_(-1, 1)

    def forward(self, x, h):
        q = torch.pow(torch.sigmoid(self.wa)+1, x+h)*torch.sigmoid(self.wk)
        qn = torch.minimum(h+x, q)
        hn = h-qn+x
        return q, hn


class SnowBucket(torch.nn.Module):
    def __init__(self, nh, nm=1, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.nm = nm
        self.w = P(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.uniform_(-1, 2)

    def forward(self, s, P, T):
        sm = F.relu(T[:, None]) * (torch.exp(self.w)+1)
        m1 = sm[:, :self.nm]
        m2 = torch.minimum(sm[:, self.nm:], s[:, self.nm:])
        m = torch.cat((m1, m2), dim=1)
        s = s-m+((T < 0)*P)[:, None]
        x = ((T > 0)*P)[:, None] + m
        return x, s


class SoilBucket(torch.nn.Module):
    def __init__(self, nh, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.wl = P(torch.Tensor(nh))
        self.we = P(torch.Tensor(nh))
        self.wk = P(torch.Tensor(nh))
        self.ws = P(torch.Tensor(nh))

    def reset_parameters(self):
        self.wl.data.uniform_(-1, 1)
        self.we.data.uniform_(-1, 1)
        self.wk.data.uniform_(-1, 1)
        self.ws.data.uniform_(-1, 1)

    def forward(self, x, h, E):
        hn = h+x
        h1 = torch.relu(hn-torch.exp(self.wl*2))
        q1 = torch.relu(h1-E[:, None]*torch.sigmoid(self.we))
        h2 = hn-h1
        q2 = h2 * torch.sigmoid(self.wk)
        h = h2-q2
        q2a = q2*torch.sigmoid(self.ws)
        q2b = q2*(1-torch.sigmoid(self.ws))
        return q1, q2a, q2b, h


class WaterNetModel2(torch.nn.Module):
    def __init__(self, nh, *, nm=0):
        super().__init__()
        self.nh = nh
        self.w_o = P(torch.Tensor(nh))
        self.FB = SnowBucket(nh, nm)
        self.GB = LinearBucket(nh)
        self.SB = SoilBucket(nh)
        self.DP = nn.Dropout()
        a = F.softmax(self.DP(self.w_o), dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_o.data.uniform_(-1, 1)

    def forward(self, P, T, E, Q0=0):
        # initial states
        nt, ns = P.shape
        F = torch.zeros(nt, ns, self.nh).cuda()
        H = torch.zeros(nt, ns, self.nh).cuda()
        G = torch.zeros(nt, ns, self.nh).cuda()
        Q = torch.zeros(nt, ns).cuda()
        f = torch.zeros(ns, self.nh).cuda()
        h = torch.zeros(ns, self.nh).cuda()
        g = torch.zeros(ns, self.nh).cuda()
        for k in range(nt):
            x, f = self.FB(f, P[k, :], T[k, :])
            F[k, :, :] = f
            q1, q2, q2b, h = self.SB(x, h, E[k, :])
            q3, g = self.GB(q2b, g)
            H[k, :, :] = h
            G[k, :, :] = g
            a = F.softmax(self.DP(self.w_o), dim=0)
            # a = F.softmax(self.w_o, dim=0)
            Q[k, :] = torch.sum((q1+q2+q3)*a, dim=1)
        return Q, F, H, G


class WaterNetModel(torch.nn.Module):
    def __init__(self, nh, *, nm=0):
        super().__init__()
        self.nh = nh
        self.w_i = P(torch.Tensor(nh))
        self.w_o = P(torch.Tensor(nh))
        self.w_e = P(torch.Tensor(nh))
        # self.LN = LinearBucket(nh)
        self.LN = LinearBucket(nh)
        self.SN = SnowBucket(nh, nm)
        self.Do = nn.Dropout()
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
            # xin = x
            q, hn = self.LN(xin, h)
            H[k, :, :] = hn
            a = F.softmax(self.Do(self.w_o), dim=0)
            Q[k, :] = torch.sum(q*a, dim=1)
            s = sn
            h = hn
        return Q, H, S
