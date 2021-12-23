import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def convTS(x, w):
    nt, ns, nh = x.shape
    nr = int(w.shape[1]/nh)
    r = torch.softmax(w.view(ns*nh, 1, nr), dim=-1)
    a = x.permute(1, 2, 0).view(1, ns*nh, nt)
    y = F.conv1d(a, r, groups=ns*nh).view(ns, nh, nt-nr+1).permute(2, 0, 1)
    return y


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


class WaterNetCQ1116(torch.nn.Module):
    def __init__(self, nh, ng, nr):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.nr = nr
        # self.fc = nn.Linear(ng, nh*9)
        # self.fcT = nn.Linear(nf+ng, nh*3)

        self.fc = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*10))
        self.fcR = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*nr))
        self.fcC = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*4))
        self.fcT1 = nn.Sequential(
            nn.Linear(1+ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*2))
        self.fcT2 = nn.Sequential(
            nn.Linear(3+ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh+1))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, xc, outQ=False):
        P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(6)]
        nt, ns = P.shape
        nh = self.nh
        nr = self.nr
        # Ta = (T1+T2)/2
        vp = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        vp[T1 >= 0] = 1
        vp[T2 <= 0] = 0
        Sv = torch.zeros(ns, nh).cuda()
        S0 = torch.zeros(ns, nh).cuda()
        S2 = torch.zeros(ns, nh).cuda()
        S3 = torch.zeros(ns, nh).cuda()
        C0 = torch.zeros(ns, nh).cuda()
        C2 = torch.zeros(ns, nh).cuda()
        C3 = torch.zeros(ns, nh).cuda()
        w = self.fc(xc)
        wR = self.fcR(xc)
        wC = self.fcC(xc)
        xcT1 = torch.cat([LAI[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
        xcT2 = torch.cat([R[:, :, None], T1[:, :, None], T2[:, :, None],
                          torch.tile(xc, [nt, 1, 1])], dim=-1)
        v1 = self.fcT1(xcT1)
        v2 = self.fcT2(xcT2)
        k1 = torch.sigmoid(w[:, nh:nh*2])
        k2 = torch.sigmoid(w[:, nh*2:nh*3])
        k23 = torch.sigmoid(w[:, nh*3:nh*4])
        k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
        gl = torch.exp(w[:, nh*5:nh*6])*2
        ga = torch.softmax(self.DP(w[:, nh*6:nh*7]), dim=1)
        qb = torch.relu(w[:, nh*7:nh*8])
        ge1 = torch.relu(w[:, nh*8:nh*9])
        ge2 = torch.relu(w[:, nh*9:nh*10])
        vi = F.hardsigmoid(v1[:, :, :nh])
        vk = F.hardsigmoid(v1[:, :, nh:nh*2])
        vm = torch.exp(v2[:, :, :nh]*2)
        r2 = torch.sigmoid(w[:, nh*0:nh*1])
        r3 = torch.sigmoid(w[:, nh*1:nh*2])
        ce2 = torch.exp(w[:, nh*2:nh*3])
        ce3 = torch.exp(w[:, nh*3:nh*4])
        # vp = F.hardsigmoid(v2[:, :, -1])
        Ps = P*(1-vp)
        Pl = P*vp
        Pl1 = Pl[:, :, None]*(1-vi)
        Pl2 = Pl[:, :, None]*vi
        Ev1 = E[:, :, None]*ge1
        Ev2 = E[:, :, None]*ge2
        Q1T = torch.zeros(nt, ns, nh).cuda()
        Q2T = torch.zeros(nt, ns, nh).cuda()
        Q3T = torch.zeros(nt, ns, nh).cuda()
        M1T = torch.zeros(nt, ns, nh).cuda()
        M2T = torch.zeros(nt, ns, nh).cuda()
        M3T = torch.zeros(nt, ns, nh).cuda()
        for k in range(nt):
            H0 = S0+Ps[k, :, None]
            qSm = torch.minimum(H0, vm[k, :, :])
            Hv = torch.relu(Sv+Pl1[k, :, :] - Ev1[k, :, :])
            qv = Sv*vk[k, :, :]
            H2 = torch.relu(S2+qSm+qv-Ev2[k, :, :]+Pl2[k, :, :])
            Q1 = torch.relu(H2-gl)**k1
            q2 = torch.minimum(H2, gl)*k2
            Q2 = q2*(1-k23)
            H3 = S3+q2*k23
            Q3 = H3*k3+qb
            S0 = H0-qSm
            Sv = Hv-qv
            S2 = H2-Q1-q2
            S3 = H3-Q3
            G2 = (C2*S2 + r2*(ce2-C2)*H2)/H2
            G3 = (C2*q2*k23 + r3*(ce3-C3)*H2)/H2
            M1T[k, :, :] = 0
            Q1T[k, :, :] = Q1
            Q2T[k, :, :] = Q2
            Q3T[k, :, :] = Q3
        r = torch.relu(wR[:, :nh*nr])
        Q1R = convTS(Q1T, r)
        Q2R = convTS(Q2T, r)
        Q3R = convTS(Q3T, r)
        yOut = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)

        if outQ:
            return yOut, (Q1R, Q2R, Q3R)
        else:
            return yOut
