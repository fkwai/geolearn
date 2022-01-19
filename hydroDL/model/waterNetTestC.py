import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from hydroDL.model.waterNet import convTS, sepPar


class Wn0110C1(torch.nn.Module):
    def __init__(self, nh, ng, nr, nc):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.nr = nr
        self.nc = nc
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
        # [vi,ve,vm]
        self.vLst = ['skip', 'relu', 'exp']
        self.fcT = nn.Sequential(
            nn.Linear(6+ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*len(self.vLst)))
        # [cp,cs,cg]
        self.cLst = ['exp', 'exp', 'exp']
        self.fcCT = nn.Linear(2+ng, nc*nh*len(self.cLst))
        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, xc):
        P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
        nt, ns = P.shape
        nh = self.nh
        nr = self.nr
        Sf = torch.zeros(ns, nh).cuda()
        Ss = torch.zeros(ns, nh).cuda()
        Sg = torch.zeros(ns, nh).cuda()
        xT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
        w = self.fcW(xc)
        [kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, self.wLst)
        gL = gL*2
        ga = torch.softmax(self.DP(ga), dim=1)
        # ga = torch.softmax(ga, dim=1)
        v = self.fcT(xT)
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
            Sg = (1-kg)*(Sg+qsg)-qb
            Q1T[k, :, :] = qp
            Q2T[k, :, :] = qso
            Q3T[k, :, :] = qg
        r = torch.relu(wR[:, :nh*nr])
        Q1R = convTS(Q1T, r)
        Q2R = convTS(Q2T, r)
        Q3R = convTS(Q3T, r)
        outQ = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)
        # concentrations
        xTc = torch.cat([T1[:, :, None], T2[:, :, None],
                        torch.tile(xc, [nt, 1, 1])], dim=-1)
        wC = self.fcTC(xTc)
        [cp, cs, cg] = sepPar(wC, nh, self.cLst)
        C1T = Q1T*cp/10
        C2T = Q2T*cs
        C3T = Q3T*cg
        C1R = convTS(C1T, r)
        C2R = convTS(C2T, r)
        C3R = convTS(C3T, r)
        outC = torch.sum((C1R+C2R+C3R)*ga, dim=2)/outQ
        return outQ, outC


class Wn0110C2(torch.nn.Module):
    def __init__(self, nh, ng, nr, nc):
        # with a interception bucket
        super().__init__()
        self.nh = nh
        self.ng = ng
        self.nr = nr
        self.nc = nc
        self.fcR = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*nr))
        # [kp, ks, kg, gp, gL, qb, ga]
        self.wLst = [
            'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
            'exp', 'relu', 'skip']
        self.fcW = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*len(self.wLst)))
        # [vi,ve,vm]
        self.vLst = ['skip', 'relu', 'exp']
        self.fcT = nn.Sequential(
            nn.Linear(6+ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*len(self.vLst)))
        # [eqs,eqg]
        self.cLst = ['exp', 'exp']
        self.fcC = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nc*nh*len(self.cLst)))
        # [rs,rg]
        self.ctLst = ['sigmoid', 'sigmoid']
        self.fcCT = nn.Linear(2+ng, nc*nh*len(self.cLst))

        self.DP = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, xc):
        P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
        nt, ns = P.shape
        nh = self.nh
        nr = self.nr
        nc = self.nc
        Sf = torch.zeros(ns, nh).cuda()
        Ss = torch.zeros(ns, nh).cuda()
        Sg = torch.zeros(ns, nh).cuda()
        w = self.fcW(xc)
        [kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, self.wLst)
        gL = gL*2
        qb = qb+1e-5
        # kg = kg/10
        ga = torch.softmax(self.DP(ga), dim=1)
        # ga = torch.softmax(ga, dim=1)
        xT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
        v = self.fcT(xT)
        [vi, ve, vm] = sepPar(v, nh, self.vLst)
        vi = F.hardsigmoid(v[:, :, :nh]*2)
        ve = ve*2
        wc = self.fcC(xc)
        [eqs, eqg] = sepPar(wc, nh*nc, self.cLst)
        eqs = eqs.view(ns, nh, nc).permute(-1, 0, 1)*10
        eqg = eqg.view(ns, nh, nc).permute(-1, 0, 1)*10
        Cs = torch.zeros(nc, ns, nh).cuda()
        Cg = eqg*qb/kg
        xTC = torch.cat([T1[:, :, None], T2[:, :, None],
                        torch.tile(xc, [nt, 1, 1])], dim=-1)
        vc = self.fcCT(xTC)
        [rs, rg] = sepPar(vc, nh*nc, self.cLst)
        rs = rs.view(nt, ns, nh, nc).permute(-1, 0, 1, 2)
        rg = rg.view(nt, ns, nh, nc).permute(-1, 0, 1, 2)
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
        C2T = torch.zeros(nc, nt, ns, nh).cuda()
        C3T = torch.zeros(nc, nt, ns, nh).cuda()
        for k in range(nt):
            qf = torch.minimum(Sf+Ps[k, :, None], vm[k, :, :])
            Sf = torch.relu(Sf+Ps[k, :, None]-vm[k, :, :])
            H = torch.relu(Ss+Pl[k, :, :]+qf-Ev[k, :, :])
            qp = torch.relu(kp*(H-gL))
            qs = ks*torch.minimum(H, gL)
            Ss = H-qp-qs
            Cs = (Cs+rs[:, k, :, :]*eqs*Ss)/(1+ks+rs[:, k, :, :])
            qsg = qs*gp
            qg = kg*(Sg+qsg)+qb
            Sg = (1-kg)*(Sg+qsg)-qb
            Hg = Sg+qb/kg
            Cg = (Cg+rg[:, k, :, :]*eqg*Hg+Cs*ks*gp)/(1+kg+rg[:, k, :, :])
            Q1T[k, :, :] = qp
            Q2T[k, :, :] = qs*(1-gp)
            Q3T[k, :, :] = qg
            C2T[:, k, :, :] = Cs*ks*(1-gp)
            C3T[:, k, :, :] = Cg*kg
        r = torch.relu(wR[:, :nh*nr])
        Q1R = convTS(Q1T, r)
        Q2R = convTS(Q2T, r)
        Q3R = convTS(Q3T, r)
        outQ = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)
        outCLst = list()
        for k in range(nc):
            C2R = convTS(C2T[k, ...], r)
            C3R = convTS(C3T[k, ...], r)
            temp = torch.sum((C2R+C3R)*ga, dim=2)/outQ
            outCLst.append(temp)
        outC = torch.stack(outCLst, dim=-1)
        return outQ, outC
