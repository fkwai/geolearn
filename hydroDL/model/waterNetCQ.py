import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class WaterNetCQ2(torch.nn.Module):
    def __init__(self, nh, ng):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.fc = nn.Linear(ng, nh*8+1)
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
        H1 = torch.zeros(ns, self.nh).cuda()
        H2 = torch.zeros(ns, self.nh).cuda()
        # I0 = torch.zeros(ns, self.nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        # w = self.fc2(torch.tanh(self.fc1(xc)))
        w = self.fc(xc)
        gm = torch.exp(w[:, :nh])+1
        ge = torch.sigmoid(w[:, nh:nh*2])*2
        k0 = torch.sigmoid(w[:, nh*3:nh*4])
        k1 = torch.sigmoid(w[:, nh*4:nh*5])
        k2 = torch.sigmoid(w[:, nh*5:nh*6])
        gl = torch.exp(w[:, nh*6:nh*7])
        ga = torch.softmax(self.DP(w[:, nh*4:nh*5]), dim=1)
        qb = torch.relu(w[:, -1])/nh
        kb = torch.sigmoid(w[:, nh*7:nh*8])/10
        vi = torch.sigmoid(w[:, nh*8:nh*9])
        for k in range(nt):
            S = S0+Ps[k, :, None]
            Sm = torch.minimum(S0, torch.relu(Ta[k, :, None]*gm))
            # I1 = I0+Pl[k, :, None]
            G1 = torch.relu(H1+Sm+Pl[k, :, None]*vi - E[k, :, None]*ge)
            G0 = torch.relu(H0+G1-gl+Pl[k, :, None]*(1-vi))
            Q0 = G0*k0
            Q1a = torch.minimum(G1, gl)*k1
            Q1 = Q1a*(1-kb)
            G2 = H2+Q1a*kb
            Q2 = G2*k2
            H0 = G0-Q0
            H1 = torch.minimum(G1-Q1a, gl)
            H2 = G2-Q2
            S0 = S-Sm
            # I0 = torch.relu(I1*(1-gi)-E[k, :, None])
            # Y = torch.sum((Q1+Q2+qb[:, None])*ga, dim=1)
            Y = torch.sum((Q0+Q1+Q2+qb[:, None])*ga, dim=1)
            Yout[k, :] = Y
        return Yout
