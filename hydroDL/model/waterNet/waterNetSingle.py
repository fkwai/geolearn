import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class RnnCell(torch.nn.Module):
    def __init__(self, *, nh):
        super().__init__()
        self.hiddenSize = nh
        self.w_i = Parameter(torch.Tensor(nh, 1))
        self.w_h = Parameter(torch.Tensor(nh, nh))
        self.w_o = Parameter(torch.Tensor(1, nh))
        self.b_i = Parameter(torch.Tensor(nh))
        self.b_h = Parameter(torch.Tensor(nh))
        self.b_o = Parameter(torch.Tensor(1))
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
        self.h0 = Parameter(torch.Tensor(nh, 1))

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
        self.w_i = Parameter(torch.Tensor(nh, 1))
        self.w_o = Parameter(torch.Tensor(1, nh))
        self.b_i = Parameter(torch.Tensor(nh))
        self.b_o = Parameter(torch.Tensor(1))
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
        self.wk = P(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.wk.data.uniform_(-1, 1)

    def forward(self, x, h):
        q = (x+h)*torch.sigmoid(self.wk)
        hn = h-q+x
        return q, hn


class LinearBucket2(torch.nn.Module):
    def __init__(self, nh, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.wk = P(torch.Tensor(nh))
        self.ws = P(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.wk.data.uniform_(-1, 1)

    def forward(self, x, h):
        q = (x+h)*torch.sigmoid(self.wk)
        q1 = q*torch.sigmoid(self.ws)
        q2 = q*(1-torch.sigmoid(self.ws))
        hn = h-q+x
        return q1, q2, hn


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
        q = torch.pow(torch.sigmoid(self.wa), x+h)*torch.sigmoid(self.wk)
        qn = torch.minimum(h+x, q)
        hn = h-qn+x
        return q, hn


class SoilBucket(torch.nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh
        self.wk1 = Parameter(torch.Tensor(nh))
        self.wk2 = Parameter(torch.Tensor(nh))
        self.wk3 = Parameter(torch.Tensor(nh))
        self.we1 = Parameter(torch.Tensor(nh))
        self.we2 = Parameter(torch.Tensor(nh))
        self.wl = Parameter(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.wk1.data.uniform_(-1, 1)
        self.wk2.data.uniform_(-1, 1)
        self.wk3.data.uniform_(-1, 1)
        self.we1.data.uniform_(-1, 1)
        self.we2.data.uniform_(-1, 1)
        self.wl.data.uniform_(-1, 1)

    def forward(self, x, h1, h2, E):
        e1 = E[:, None]*torch.sigmoid(self.we1)
        e2 = E[:, None]*torch.sigmoid(self.we2)
        L = torch.exp(self.wl)
        h1 = torch.relu(h1+h2+x-L)
        q1 = h1 * torch.sigmoid(self.wk1)
        h2 = torch.minimum(h1+h2+x, L)
        q2 = h2 * torch.sigmoid(self.wk2)
        q3 = h2 * torch.sigmoid(self.wk2)
        h1 = torch.relu(h1-q1-e1)
        h2 = torch.relu(h2-q2-e2-q3)
        return q1, q2, q3, h1, h2


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


class SnowNet(torch.nn.Module):
    def __init__(self, nh, nm=1, *, wR=(0, 1)):
        super().__init__()
        self.nh = nh
        self.nm = nm
        self.w = Parameter(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.uniform_(-1, 2)

    def forward(self, s, P, T1, T2):
        # split: 1-np.arccos((t2+t1)/(t2-t1))/np.pi
        rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        rP[T1 >= 0] = 1
        rP[T2 <= 0] = 0
        s = s+((1-rP)*P)[:, None]
        sm = F.relu((T1[:,  None]+T2[:,  None])/2) * (torch.exp(self.w)+1)
        m1 = sm[:, :self.nm]
        m2 = torch.minimum(sm[:, self.nm:], s[:, self.nm:])
        m = torch.cat((m1, m2), dim=1)
        s = s-m
        x = (rP*P)[:, None] + m
        return x, s


class SoilModel(torch.nn.Module):
    def __init__(self, nh, *, nm=0):
        super().__init__()
        self.nh = nh
        self.wo = Parameter(torch.Tensor(nh))
        self.Snow = SnowNet(nh, nm)
        self.Soil = SoilBucket(nh)
        self.GW = LinearBucket(nh)
        self.DP = nn.Dropout()
        self.b = Parameter(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.wo.data.uniform_(-1, 1)
        self.b.data.uniform_(0, 1)

    def forward(self, P, T1, T2, E, Q0=0):
        # initial states
        nt, ns = P.shape
        S = torch.zeros(nt, ns, self.nh).cuda()
        H1 = torch.zeros(nt, ns, self.nh).cuda()
        H2 = torch.zeros(nt, ns, self.nh).cuda()
        Q = torch.zeros(nt, ns).cuda()
        Qs = torch.zeros(nt, ns, 3).cuda()
        s = torch.zeros(ns, self.nh).cuda()
        h1 = torch.zeros(ns, self.nh).cuda()
        h2 = torch.zeros(ns, self.nh).cuda()
        h3 = torch.zeros(ns, self.nh).cuda()

        a = torch.softmax(self.DP(self.wo), dim=0)
        # a = torch.softmax(self.wo, dim=0)
        for k in range(nt):
            x, s = self.Snow(s, P[k, :], T1[k, :], T2[k, :])
            S[k, :, :] = s
            q1, q2, q2b, h1, h2 = self.Soil(x, h1, h2, E[k, :])
            H1[k, :, :] = h1
            H2[k, :, :] = h2
            q3, h3 = self.GW(q2b, h3)
            Q[k, :] = torch.sum((q1+q2+q3+torch.relu(self.b))*a, dim=1)
            Qs[k, :, 0] = torch.sum(q1*a, dim=1)
            Qs[k, :, 1] = torch.sum(q2*a, dim=1)
            Qs[k, :, 2] = torch.sum(q2*a, dim=1)

        return Q, (S, H1, H2, Qs)


class SoilModelCQ(torch.nn.Module):
    def __init__(self, nh, *, nm=0):
        super().__init__()
        self.nh = nh
        self.wo = Parameter(torch.Tensor(nh))
        self.Snow = SnowBucket(nh, nm)
        self.Soil = SoilBucket(nh)
        self.DP = nn.Dropout()
        self.b = Parameter(torch.Tensor(nh))
        self.wc = Parameter(torch.Tensor(nh))
        self.R = Parameter(torch.Tensor(nh))
        self.reset_parameters()

    def reset_parameters(self):
        self.wo.data.uniform_(-1, 1)
        self.b.data.uniform_(0, 1)
        self.R.data.uniform_(0, 1)
        self.wc.data.uniform_(-1, 1)

    def forward(self, P, T, E, Q0=0):
        # initial states
        nt, ns = P.shape
        S = torch.zeros(nt, ns, self.nh).cuda()
        H1 = torch.zeros(nt, ns, self.nh).cuda()
        H2 = torch.zeros(nt, ns, self.nh).cuda()
        Q = torch.zeros(nt, ns).cuda()
        C = torch.zeros(nt, ns).cuda()
        Qs = torch.zeros(nt, ns, 3).cuda()
        s = torch.zeros(ns, self.nh).cuda()
        h1 = torch.zeros(ns, self.nh).cuda()
        h2 = torch.zeros(ns, self.nh).cuda()
        a = torch.softmax(self.DP(self.wo), dim=0)
        for k in range(nt):
            x, s = self.Snow(s, P[k, :], T[k, :])
            S[k, :, :] = s
            q1, q2, h1, h2 = self.Soil(x, h1, h2, E[k, :])
            H1[k, :, :] = h1
            H2[k, :, :] = h2
            Q[k, :] = torch.sum((q1+q2)*a, dim=1)
            Qs[k, :, 0] = torch.sum(q1*a, dim=1)
            Qs[k, :, 1] = torch.sum(q2*a, dim=1)
            tc = 1+torch.exp(self.wc)
            r = 1/torch.sigmoid(self.Soil.wk2)/tc
            c = torch.mean(q2*a*r+1e-5, dim=1)/(Q[k, :]+1e-5)
            C[k, :] = c
        return Q, C, (S, H1, H2, Qs)


class WaterNetModel(torch.nn.Module):
    def __init__(self, nh, *, nm=0):
        super().__init__()
        self.nh = nh
        self.w_i = Parameter(torch.Tensor(nh))
        self.w_o = Parameter(torch.Tensor(nh))
        self.w_c = Parameter(torch.Tensor(nh))
        # self.LN = LinearBucket(nh)
        self.LN = LinearBucket(nh)
        self.SN = SnowBucket(nh, nm)
        self.Do = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        self.w_i.data.uniform_(-1, 1)
        self.w_o.data.uniform_(-1, 1)
        self.w_c.data.uniform_(-1, 1)

    def forward(self, P, T, Q0=0):
        # initial states
        nt, ns = P.shape
        S = torch.zeros(nt, ns, self.nh).cuda()
        H = torch.zeros(nt, ns, self.nh).cuda()
        Q = torch.zeros(nt, ns).cuda()
        C = torch.zeros(nt, ns).cuda()
        s = torch.zeros(ns, self.nh).cuda()
        h = torch.zeros(ns, self.nh).cuda()
        for k in range(nt):
            x, sn = self.SN(s, P[k, :], T[k, :])
            S[k, :, :] = sn
            xin = x*torch.sigmoid(self.w_i)
            q, hn = self.LN(xin, h)
            H[k, :, :] = hn
            a = F.softmax(self.Do(self.w_o), dim=0)
            Q[k, :] = torch.sum(q*a, dim=1)
            s = sn
            h = hn
            tc = 1+torch.exp(self.w_c)
            r = 1/torch.sigmoid(self.LN.wk)/tc
        return Q, H, S


class WaterNet1(torch.nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh
        self.w = Parameter(torch.Tensor(nh*4))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        # std = 1.0 / math.sqrt(self.nh)
        for w in self.parameters():
            w.data.uniform_(-1, 1)

    def forward(self, P, T1, T2, E):
        # initial states
        nt, ns = P.shape
        nh = self.nh
        w = self.w
        Ta = (T1+T2)/2
        rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        rP[T1 >= 0] = 1
        rP[T2 <= 0] = 0
        Ps = (1-rP)*P
        Pl = rP*P
        S0 = torch.zeros(ns, self.nh).cuda()
        H0 = torch.zeros(ns, self.nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        gm = torch.exp(w[:nh])+1
        ge = torch.sigmoid(w[nh:nh*2])
        go = torch.sigmoid(w[nh*2:nh*3])
        ga = torch.softmax(self.DP(w[nh*3:]), dim=0)
        for k in range(nt):
            Sm = torch.minimum(S0, torch.relu(Ta[k, :, None]*gm))
            S = S0+Ps[k, :, None]
            H = torch.relu(H0+Sm+Pl[k, :, None] - E[k, :, None]*ge)
            # H = torch.relu(H0+(Sm+Pl[k, :, None])*ge)
            Q = H*go
            H0 = H-Q
            S0 = S-Sm
            Y = torch.sum(Q*ga, dim=1)
            Yout[k, :] = Y
        return Yout
