import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from hydroDL.model.waterNet import convTS, sepPar
from hydroDL.model import waterNetTest, waterNet


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


class Trans0110C2(waterNetTest.WaterNet0110):
    def __init__(self, nh, ng, nr, nc, dictQ=None, freeze=False):
        super().__init__(nh, ng, nr)
        if dictQ is not None:
            super().load_state_dict(dictQ)
            if freeze:
                for layer in self.children():
                    for param in layer.parameters():
                        param.requires_grad = False
        self.nc = nc
        # [eqs,eqg]
        self.cLst = ['exp', 'exp']
        self.fcC = nn.Sequential(
            nn.Linear(self.ng, 256),
            nn.Dropout(),
            nn.Linear(256, nc*nh*len(self.cLst)))
        # [rs,rg]
        self.ctLst = ['sigmoid', 'sigmoid']
        self.fcCT = nn.Linear(2+self.ng, nc*nh*len(self.cLst))
        # ISSUE - no reset parameters - and the old one not work neither

    def forward(self, x, xc):
        P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
        nt, ns = P.shape
        nh = self.nh
        nr = self.nr
        nc = self.nc
        Sf = torch.zeros(ns, nh)
        Ss = torch.zeros(ns, nh)
        Sg = torch.zeros(ns, nh)
        Cs = torch.zeros(nc, ns, nh)
        if torch.cuda.is_available():
            Sf = Sf.cuda()
            Ss = Ss.cuda()
            Sg = Sg.cuda()
            Cs = Cs.cuda()
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
        Q1T = torch.zeros(nt, ns, nh)
        Q2T = torch.zeros(nt, ns, nh)
        Q3T = torch.zeros(nt, ns, nh)
        C2T = torch.zeros(nc, nt, ns, nh)
        C3T = torch.zeros(nc, nt, ns, nh)
        if torch.cuda.is_available():
            Q1T=Q1T.cuda()
            Q2T=Q2T.cuda()
            Q3T=Q3T.cuda()
            C2T=C2T.cuda()
            C3T=C3T.cuda()
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


class Wn0119C2(waterNet.WaterNet0119):
    def __init__(self, nh, ng, nr, nc, dictQ=None, freeze=False):
        super().__init__(nh, ng, nr)
        if dictQ is not None:
            super().load_state_dict(dictQ)
            if freeze:
                for layer in self.children():
                    for param in layer.parameters():
                        param.requires_grad = False
        self.nc = nc
        # [eqs,eqg]
        self.cLst = ['exp', 'exp']
        self.fcC = nn.Sequential(
            nn.Linear(self.ng, 256),
            nn.Dropout(),
            nn.Linear(256, nc*nh*len(self.cLst)))
        # [rs,rg]
        self.ctLst = ['sigmoid', 'sigmoid']
        self.fcCT = nn.Linear(2+self.ng, nc*nh*len(self.cLst))


class Wn0119EMsolo(torch.nn.Module):
    def __init__(self, nh, nr, nc, nm):
        super().__init__()
        self.nh = nh
        self.nr = nr
        self.nc = nc
        self.DP = nn.Dropout()
        self.wR = Parameter(torch.randn(nh*nr).cuda())
        self.wLst = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
                     'exp', 'relu', 'skip']
        self.w = Parameter(torch.randn(nh*len(self.wLst)).cuda())
        self.vLst = ['skip', 'relu', 'exp']
        self.fcT = nn.Sequential(
            nn.Linear(6, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nh*len(self.vLst))).cuda()
        # self.cp = Parameter(torch.rand(nh, nc).cuda())
        # self.cs = Parameter(torch.rand(nh, nc).cuda())
        # self.cg = Parameter(torch.rand(nh, nc).cuda())
        self.cp = Parameter(torch.rand(nm, nc).cuda())
        self.cs = Parameter(torch.rand(nm, nc).cuda())
        self.cg = Parameter(torch.rand(nm, nc).cuda())
        self.nm = nm

    def forward(self, x, outQ=False):
        nh = self.nh
        [kp, ks, kg, gp, gL, qb, ga] = sepPar(self.w, self.nh, self.wLst)
        gL = gL**2
        kg = kg/10
        # ga = torch.softmax(self.DP(ga), dim=-1)
        ga = torch.softmax(ga, dim=-1)
        v = self.fcT(x)
        [vi, ve, vm] = sepPar(v, self.nh, self.vLst)
        vi = F.hardsigmoid(vi*2)
        ve = ve*2
        rf = torch.relu(self.wR)
        P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
        nt, ns = P.shape
        Sf = torch.zeros(ns, nh).cuda()
        Ss = torch.zeros(ns, nh).cuda()
        Sg = torch.zeros(ns, nh).cuda()
        Ps, Pl, Ev = WaterNet0119.forwardPreQ(P, E, T1, T2, vi, ve)
        QpT = torch.zeros(nt, ns, nh).cuda()
        QsT = torch.zeros(nt, ns, nh).cuda()
        QgT = torch.zeros(nt, ns, nh).cuda()
        for k in range(nt):
            qp, qs, qg, Sf, Ss, Sg = WaterNet0119.forwardStepQ(
                Sf, Ss, Sg, Ps[k, :, None], Pl[k, :, :],
                Ev[k, :, :], vm[k, :, :], kp, ks, kg, gL, gp, qb)
            QpT[k, :, :] = qp
            QsT[k, :, :] = qs
            QgT[k, :, :] = qg
        QpR = convTS(QpT, rf)*ga
        QsR = convTS(QsT, rf)*ga
        QgR = convTS(QgT, rf)*ga
        Qout = torch.sum(QpR+QsR+QgR, dim=-1)
        cp = torch.relu(torch.exp(self.cp)-1)
        cs = torch.relu(torch.exp(self.cs)-1)
        cg = torch.relu(torch.exp(self.cg)-1)
        cp = cp.repeat(int(nh/self.nm), 1)
        cs = cs.repeat(int(nh/self.nm), 1)
        cg = cg.repeat(int(nh/self.nm), 1)
        CpR = torch.matmul(QpR/Qout[:, :, None], cp)
        CsR = torch.matmul(QsR/Qout[:, :, None], cs)
        CgR = torch.matmul(QgR/Qout[:, :, None], cg)
        Cout = CpR+CsR+CgR
        # Cout = torch.sum((QpR*self.cp+QsR*self.cs+QgR*self.cg)*ga, dim=-1)
        yOut = torch.cat([Qout[..., None], Cout], dim=-1)
        if outQ is True:
            return yOut, (QpR, QsR, QgR)
        else:
            return yOut


class Wn0119EM(waterNetTest.WaterNet0119):
    def __init__(self, nh, ng, nr, nc, nm):
        super().__init__(nh, ng, nr)
        self.cp = Parameter(torch.rand(nm, nc).cuda())
        self.cs = Parameter(torch.rand(nm, nc).cuda())
        self.cg = Parameter(torch.rand(nm, nc).cuda())
        self.nc = nc
        self.nm = nm
        self.fcC = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            # nn.Dropout(),
            nn.Linear(256, nm*nc*3)).cuda()
        self.cLst = ['skip', 'skip', 'skip']
        self.reset_parameters()

    def forward(self, x, xc, outStep=False):
        Qout, (QpR, QsR, QgR), (SfT, SsT, SgT) = super().forward(
            x, xc, outStep=True)
        c = self.fcC(xc)
        nh = self.nh
        nm = self.nm
        nc = self.nc
        [cpT, csT, cgT] = sepPar(c, self.nm*self.nc, self.cLst)
        cp = cpT.view(-1, nm, nc)
        cs = csT.view(-1, nm, nc)
        cg = cgT.view(-1, nm, nc)
        # cp = torch.relu(torch.exp(cp))
        # cs = torch.relu(torch.exp(cs))
        # cg = torch.relu(torch.exp(cg))
        cp = torch.exp(cp*10)
        cs = torch.exp(cs*10)
        cg = torch.exp(cg*10)
        cp = cp.repeat(1, int(nh/self.nm), 1)
        cs = cs.repeat(1, int(nh/self.nm), 1)
        cg = cg.repeat(1, int(nh/self.nm), 1)
        nt = Qout.shape[0]
        # QpP = torch.nan_to_num(QpR/Qout[:, :, None])
        # QsP = torch.nan_to_num(QsR/Qout[:, :, None])
        # QgP = torch.nan_to_num(QgR/Qout[:, :, None])        
        QpP = QpR/Qout[:, :, None]
        QsP = QsR/Qout[:, :, None]
        QgP = QgR/Qout[:, :, None]
        CpR = QpP[:, :, :, None] * cp.repeat(nt, 1, 1, 1)
        CsR = QsP[:, :, :, None] * cs.repeat(nt, 1, 1, 1)
        CgR = QgP[:, :, :, None] * cg.repeat(nt, 1, 1, 1)
        Cout = torch.sum(CpR+CsR+CgR, dim=2)
        if torch.isnan(Cout).sum() > 0:
            print('nan in c')
        if torch.isnan(Qout).sum() > 0:
            print('nan in q')
        # Cout = torch.sum((QpR*self.cp+QsR*self.cs+QgR*self.cg)*ga, dim=-1)
        yOut = torch.cat([Qout[..., None], Cout], dim=-1)
        if outStep is True:
            return yOut, (QpR, QsR, QgR), (SfT, SsT, SgT), (cp, cs, cg)
        else:
            return yOut
