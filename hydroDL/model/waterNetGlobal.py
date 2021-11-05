import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class WaterNet1(torch.nn.Module):
    def __init__(self, nh, ng):
        super().__init__()
        self.nh = nh
        self.ng = ng

        self.fc1 = nn.Linear(ng, 256)
        self.fc2 = nn.Linear(256, nh*4)
        self.fc3 = nn.Linear(ng, nh*4)
        self.norm = nn.LayerNorm(nh*4)

        self.w = Parameter(torch.Tensor(nh*4))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nh)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, xc):
        P, E, T1, T2 = [x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]]
        nt, ns = P.shape
        nh = self.nh
        Ta = (T1+T2)/2
        rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        rP[T1 >= 0] = 1
        rP[T2 <= 0] = 0
        Ps = (1-rP)*P
        Pl = rP*P
        S0 = torch.zeros(ns, self.nh).cuda()
        H0 = torch.zeros(ns, self.nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        w = self.fc3(xc)
        gm = torch.exp(w[:, :nh])+1
        ge = torch.sigmoid(w[:, nh:nh*2])
        go = torch.sigmoid(w[:, nh*2:nh*3])
        ga = torch.softmax(self.DP(w[:, nh*3:]), dim=1)
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


class WaterNet2(torch.nn.Module):
    def __init__(self, nh, ng):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.fc = nn.Linear(ng, nh*8+2)
        self.w = Parameter(torch.Tensor(nh*8+2))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nh)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, xc):
        P, E, T1, T2 = [x[:, :, k] for k in range(4)]
        nt, ns = P.shape
        nh = self.nh
        Ta = (T1+T2)/2
        rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        rP[T1 >= 0] = 1
        rP[T2 <= 0] = 0
        Ps = (1-rP)*P
        Pl = rP*P
        S0 = torch.zeros(ns, self.nh).cuda()
        H0 = torch.zeros(ns, self.nh).cuda()
        G0 = torch.zeros(ns, self.nh).cuda()
        # I0 = torch.zeros(ns, self.nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        w = self.fc(xc)
        gm = torch.exp(w[:, :nh])+1
        ge = torch.sigmoid(w[:, nh:nh*2])*2
        go = torch.sigmoid(w[:, nh*2:nh*3])
        gl = torch.exp(w[:, nh*3:nh*4]*2)
        ga = torch.softmax(self.DP(w[:, nh*4:nh*5]), dim=1)
        gb = torch.sigmoid(w[:, nh*5:nh*6])
        qb = torch.relu(w[:, -1])/nh
        kb = torch.sigmoid(w[:, nh*6:nh*7])/10
        gi = torch.sigmoid(w[:, nh*7:nh*8])

        for k in range(nt):
            S = S0+Ps[k, :, None]
            Sm = torch.minimum(S0, torch.relu(Ta[k, :, None]*gm))
            # I1 = I0+Pl[k, :, None]
            H = torch.relu(H0+Sm+Pl[k, :, None]*gi - E[k, :, None]*ge)
            Q1 = torch.relu(H-gl)
            Q2a = torch.minimum(H, gl)*go
            Q2 = Q2a*(1-gb)
            G = G0+Q2a*gb
            Q3 = G*kb
            H0 = torch.minimum(H-Q2a, gl)
            G0 = G-Q3
            S0 = S-Sm
            # I0 = torch.relu(I1*(1-gi)-E[k, :, None])
            # Y = torch.sum((Q1+Q2+qb[:, None])*ga, dim=1)
            Y = torch.sum((Q1+Q2+Q3+qb[:, None])*ga, dim=1)
            Yout[k, :] = Y
        return Yout


class WaterNet3(torch.nn.Module):
    def __init__(self, nh, nf, ng):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.fc = nn.Linear(ng, nh*8+1)
        # self.fc = nn.Sequential(
        #     nn.Linear(ng, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, nh*8+1))
        self.fcT = nn.Linear(nf+ng, nh)
        # self.fcT = nn.Sequential(
        #     nn.Linear(nf+ng, 256),
        #     # nn.Tanh(),
        #     nn.Linear(256, nh*8+1))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, xc, outQ=False):
        P, E, T1, T2, LAI = [x[:, :, k] for k in range(5)]
        nt, ns = P.shape
        nh = self.nh
        Ta = (T1+T2)/2
        rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        rP[T1 >= 0] = 1
        rP[T2 <= 0] = 0
        Ps = (1-rP)*P
        Pl = rP*P
        S0 = torch.zeros(ns, nh).cuda()
        # S1 = torch.zeros(ns, self.nh).cuda()
        S2 = torch.zeros(ns, nh).cuda()
        S3 = torch.zeros(ns, nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        w = self.fc(xc)
        xcT = torch.cat([LAI[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
        v = self.fcT(xcT)
        gm = torch.exp(w[:, :nh])+1
        ge = torch.sigmoid(w[:, nh:nh*2])*2
        k2 = torch.sigmoid(w[:, nh*2:nh*3])
        k23 = torch.sigmoid(w[:, nh*3:nh*4])
        k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
        gl = torch.exp(w[:, nh*5:nh*6])*2
        ga = torch.softmax(self.DP(w[:, nh*6:nh*7]), dim=1)
        qb = torch.relu(w[:, nh*7:nh*8])
        vi = torch.sigmoid(v[:, :, :nh])
        if outQ:
            Q1out = torch.zeros(nt, ns, nh).cuda()
            Q2out = torch.zeros(nt, ns, nh).cuda()
            Q3out = torch.zeros(nt, ns, nh).cuda()

        for k in range(nt):
            H0 = S0+Ps[k, :, None]
            qSm = torch.minimum(H0, torch.relu(Ta[k, :, None]*gm))
            qIn = qSm+Pl[k, :, None]*vi[k, :, :] - E[k, :, None]*ge
            Q1 = torch.relu(S2+qIn-gl)
            H2 = torch.minimum(torch.relu(S2+qIn), gl)
            q2 = H2*k2
            Q2 = q2*(1-k23)
            H3 = S3+q2*k23
            Q3 = torch.relu(H3*k3+qb)
            S0 = H0-qSm
            S2 = H2-Q2
            S3 = H3-Q3
            Y = torch.sum((Q1+Q2+Q3)*ga, dim=1)
            Yout[k, :] = Y
            if outQ:
                Q1out[k, :, :] = Q1
                Q2out[k, :, :] = Q2
                Q3out[k, :, :] = Q3
        if outQ:
            return Yout, (Q1out, Q2out, Q3out)
        else:
            return Yout
