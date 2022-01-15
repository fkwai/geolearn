import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from hydroDL.model.waterNet import convTS, sepPar


class WaterNetSingle(torch.nn.Module):
    def __init__(self, nh):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.w = Parameter(torch.Tensor(nh*9))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.uniform_(-0.5, 1/self.nh)

    def forward(self, x):
        P, E, T1, T2 = [x[:, :, k] for k in range(4)]
        nt, ns = P.shape
        nh = self.nh
        Ta = (T1+T2)/2
        rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
        rP[T1 >= 0] = 1
        rP[T2 <= 0] = 0
        Ps = (1-rP)*P
        Pl = rP*P
        S0 = torch.zeros(ns, nh).cuda()
        S1 = torch.zeros(ns, nh).cuda()
        S2 = torch.zeros(ns, nh).cuda()
        S3 = torch.zeros(ns, nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        w = self.w[None, :]
        gm = torch.exp(w[:, :nh])+1
        ge = torch.sigmoid(w[:, nh:nh*2])*2
        k2 = torch.sigmoid(w[:, nh*2:nh*3])
        k23 = torch.sigmoid(w[:, nh*3:nh*4])
        k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
        gl = torch.exp(w[:, nh*5:nh*6])*2
        ga = torch.softmax(self.DP(w[:, nh*6:nh*7]), dim=1)
        qb = torch.relu(w[:, nh*7:nh*8])
        vi = torch.relu(w[:, nh*8:nh*9])
        for k in range(nt):
            H0 = S0+Ps[k, :, None]
            qSm = torch.minimum(H0, torch.relu(Ta[k, :, None]*gm))
            qIn = qSm+Pl[k, :, None]*vi - E[k, :, None]*ge
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
        else:
            return Yout


class WaterNet1104(torch.nn.Module):
    def __init__(self, nh, nf, ng):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        # self.fc = nn.Linear(ng, nh*9)
        # self.fcT = nn.Linear(nf+ng, nh*3)

        self.fc = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Linear(256, nh*9))
        self.fcT = nn.Sequential(
            nn.Linear(nf+ng, 256),
            nn.Tanh(),
            nn.Linear(256, nh*3))
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
        Ps = P*(1-rP)
        Pl = P*rP
        S0 = torch.zeros(ns, nh).cuda()
        S1 = torch.zeros(ns, nh).cuda()
        Sv = torch.zeros(ns, nh).cuda()
        S2 = torch.zeros(ns, nh).cuda()
        S3 = torch.zeros(ns, nh).cuda()
        Yout = torch.zeros(nt, ns).cuda()
        w = self.fc(xc)
        xcT = torch.cat([LAI[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
        v = self.fcT(xcT)
        gm = torch.exp(w[:, :nh])+1
        k1 = torch.sigmoid(w[:, nh:nh*2])
        k2 = torch.sigmoid(w[:, nh*2:nh*3])
        k23 = torch.sigmoid(w[:, nh*3:nh*4])
        k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
        gl = torch.exp(w[:, nh*5:nh*6])*2
        ga = torch.softmax(self.DP(w[:, nh*6:nh*7]), dim=1)
        qb = torch.relu(w[:, nh*7:nh*8])
        ge = torch.sigmoid(w[:, nh*8:nh*9])*5
        vi = torch.sigmoid(v[:, :, :nh])
        vk = torch.sigmoid(v[:, :, nh:nh*2])
        ve = torch.sigmoid(v[:, :, nh*2:nh*3])*5
        if outQ:
            Q1out = torch.zeros(nt, ns, nh).cuda()
            Q2out = torch.zeros(nt, ns, nh).cuda()
            Q3out = torch.zeros(nt, ns, nh).cuda()
        Pl1 = Pl[:, :, None]*(1-vi)
        Pl2 = Pl[:, :, None]*vi
        Ev = E[:, :, None]*ve
        for k in range(nt):
            H0 = S0+Ps[k, :, None]
            qSm = torch.minimum(H0, torch.relu(Ta[k, :, None]*gm))
            Hv = torch.relu(Sv+Pl1[k, :, :] - Ev[k, :, :])
            qv = Hv*vk[k, :, :]
            H2 = torch.relu(S2+qSm+qv-E[k, :, None]*ge+Pl2[k, :, :])
            Q1 = torch.relu(H2-gl)**k1
            q2 = torch.minimum(H2, gl)*k2
            Q2 = q2*(1-k23)
            H3 = S3+q2*k23
            Q3 = H3*k3+qb
            S0 = H0-qSm
            Sv = Hv-qv
            S2 = H2-Q1-q2
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


class WaterNet1116(torch.nn.Module):
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
        # S1 = torch.zeros(ns, nh).cuda()
        S0 = torch.zeros(ns, nh).cuda()
        S2 = torch.zeros(ns, nh).cuda()
        S3 = torch.zeros(ns, nh).cuda()
        w = self.fc(xc)
        wR = self.fcR(xc)
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


class WaterNet0110(torch.nn.Module):
    def __init__(self, nh, ng, nr):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.nr = nr
        self.fcR = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*nr))
        # [kp, ks, kg, gp, gl, qb, ga]
        self.wLst = [
            'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
            'exp', 'relu', 'skip']
        self.fcW = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*len(self.wLst)))
        # # [vi,ve]
        # self.v1Lst = ['skip', 'relu']
        # self.fcT1 = nn.Sequential(
        #     nn.Linear(1+ng, 256),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(256, nh*len(self.v1Lst)))
        # # [vm]
        # self.v2Lst = ['exp']
        # self.fcT2 = nn.Sequential(
        #     nn.Linear(3+ng, 256),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(256, nh+1))
        # [vi,ve,vm]
        self.vLst = ['skip', 'relu', 'exp']
        self.fcT = nn.Sequential(
            nn.Linear(6+ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*len(self.vLst)))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, xc, outQ=False):
        P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
        nt, ns = P.shape
        nh = self.nh
        nr = self.nr
        Sf = torch.zeros(ns, nh).cuda()
        Ss = torch.zeros(ns, nh).cuda()
        Sg = torch.zeros(ns, nh).cuda()
        xcT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
        # xcT1 = torch.cat([LAI[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
        # xcT2 = torch.cat([R[:, :, None], T1[:, :, None], T2[:, :, None],
        #                   torch.tile(xc, [nt, 1, 1])], dim=-1)
        w = self.fcW(xc)
        [kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, self.wLst)
        gL = gL*2
        kg = kg/10
        ga = torch.softmax(self.DP(ga), dim=1)

        # v1 = self.fcT1(xcT1)
        # [vi, ve] = sepPar(v1, nh, self.v1Lst)
        # vi = F.hardsigmoid(v1[:, :, :nh]*2)
        # ve = ve*2
        # v2 = self.fcT2(xcT2)
        # [vm] = sepPar(v2, nh, self.v2Lst)
        v = self.fcT(xcT)
        [vi, ve, vm] = sepPar(v, nh, self.vLst)
        vi = F.hardsigmoid(v[:, :, :nh]*2)
        ve = ve*2
        wR = self.fcR(xc)
        vf = torch.arccos((T1+T2)/(T2-T1))/3.1415
        vf[T1 >= 0] = 0
        vf[T2 <= 0] = 1
        Ps = P*vf
        Pla = P*(1-vf)
        Pl = Pla[:, :, None]*vi
        Ev = E[:, :, None]*ve
        Q1T = torch.zeros(nt, ns, nh).cuda()
        Q2T = torch.zeros(nt, ns, nh).cuda()
        Q3T = torch.zeros(nt, ns, nh).cuda()
        for k in range(nt):
            qf = torch.minimum(Sf+Ps[k, :, None], vm[k, :, :])
            Sf = torch.relu(Sf+Ps[k, :, None]-vm[k, :, :])
            H = torch.relu(Ss+Pl[k, :, :]+qf-Ev[k, :, :])
            qp = torch.relu(kp*(H-gL))
            qs = ks*torch.minimum(H, gL)
            Ss = H-qp-qs
            qso = qs*(1-gp)
            qsg = qs*gp
            qg = kg*(Sg+qsg)+qb
            Sg = Sg-qg
            Q1T[k, :, :] = qp
            Q2T[k, :, :] = qso
            Q3T[k, :, :] = qg
        r = torch.relu(wR[:, :nh*nr])
        Q1R = convTS(Q1T, r)
        Q2R = convTS(Q2T, r)
        Q3R = convTS(Q3T, r)
        yOut = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)
        if outQ:
            return yOut, (Q1R, Q2R, Q3R)
        else:
            return yOut
