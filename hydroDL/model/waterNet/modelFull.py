import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, trans


class WaterNet0313(torch.nn.Module):
    def __init__(self, nh, ng, nr, rho=(5, 365, 639)):
        super(WaterNet0313, self).__init__()
        self.nh = nh
        self.ng = ng
        self.nr = nr
        self.rho_short, self.rho_long, self.rho_warmup = rho
        self.paramModel(nh, ng, nr)

    def paramModel(self, nh, ng, nr, hs=256, dr=0.5):
        # rounting
        self.fcR = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(p=dr),
            nn.Linear(256, nh * nr),
        )
        # [kp, ks, kg, gp, gl, qb, ga]
        self.wLst = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'exp', 'relu', 'skip']
        self.fcW = nn.Sequential(
            nn.Linear(ng, 256),
            nn.Tanh(),
            nn.Dropout(p=dr),
            nn.Linear(256, nh * len(self.wLst)),
        )
        # [vi,ve,vm]
        self.vLst = ['skip', 'relu', 'exp']
        self.fcT = nn.Sequential(
            nn.Linear(6 + ng, 256),
            nn.Tanh(),
            nn.Dropout(p=dr),
            nn.Linear(256, nh * len(self.vLst)),
        )
        self.DP = nn.Dropout(p=dr)
        self.reset_parameters(self)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def initState(self, ns):
        nh = self.nh
        Sf = torch.zeros(ns, nh)
        Ss = torch.zeros(ns, nh)
        Sg = torch.zeros(ns, nh)
        if torch.cuda.is_available():
            Sf = Sf.cuda()
            Ss = Ss.cuda()
            Sg = Sg.cuda()
        return Sf, Ss, Sg

    def getParamK(self, xc):
        pass

    def getParamG(self, x, xc):
        pass

    def getParam(self, x, xc):
        pass

    def forwardLong(self, S, I, paramK):
        Sf, qf = bucket.snow(S, I, paramK)
        return Sf, qf

    def forwardShort(self, S, I, paramK, paramG):
        Sf0, Ss0, Sd0 = S
        Ps, Pl, Ev = I
        fm, fi = paramK
        [kp, ks, kg, gr, gL, qb] = paramG
        Ss, qp, qs = bucket.shallow(Ss0, Pl * fi + qf - Ev, (gL, kp, ks))
        Sd, qd = bucket.deep(Sd0, qs * gr, (kg, qb))
        return (Ss, Sd), (qp, qs * (1 - gr), qd)

    def forward(self, x, xc, outStep=False):
        nt = x.shape[0]
        ns = x.shape[1]
        Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
        Sf, Ss, Sd = self.initState(ns)
        g_short, k_short, k_long = self.getParam()
        Ps, Pl = trans.divideP(Prcp, T1, T2)

        for iT in range(self.rho_warmup):
            with torch.no_grad():
                Sf, qf = self.forwardLong(Sf, Ps[iT, ...], fm[iT, ...])
                (Ss, Sd), (qp, qs, qd) = self.forwardShort(
                    (Ss, Sd), (Pl, Ev), k_short, g_short
                )
        H = [Ss.detach(), Sd.detach()]
        Qp, Qs, Qd, Qf = [], [], []
        for iT in range(self.rho_warmup, nt):
            Sf, qf = self.forwardLong(Sf, Ps[iT, ...], fm[iT, ...])
            Qf.append(qf)
            if iT - self.rho_warmup % self.rho_long == 0:
                Sf.detach_()
        for iT in range(self.rho_warmup, nt):
            if iT < self.rho_warmup + self.rho_short:
                (Ss, Sd), (qp, qs, qd) = self.forwardShort(
                    (Ss, Sd), (Pl, Ev), k_short, g_short
                )
            else:
                Ss, Sd = H[0]
                for i in range(self.rho_short):
                    (Ss, Sd), (qp, qs, qd) = self.forwardShort(
                        (Ss, Sd), (Pl, Ev), k_short, g_short
                    )
                    _ = H.pop(0)
            H.append([Ss.detach(), Sd.detach()])
            Q.append([qp, qs, qd])
