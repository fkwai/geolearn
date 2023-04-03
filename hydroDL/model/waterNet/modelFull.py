import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, func
from hydroDL.model.waterNet.func import convTS, sepParam
from hydroDL.model.dropout import createMask, DropMask
from collections import OrderedDict
import time


def defineFCNN(nx, ny, hs=256, dr=0.5):
    model = nn.Sequential(
        nn.Linear(nx, hs), nn.Tanh(), nn.Dropout(p=dr), nn.Linear(hs, ny)
    )
    return model


class WaterNet0313(torch.nn.Module):
    def __init__(self, nf, ng, nh, nr, rho=(5, 365, 639), hs=256, dr=0.5):
        super(WaterNet0313, self).__init__()
        self.nf = nf
        self.nh = nh
        self.ng = ng
        self.nr = nr
        self.hs = hs
        self.dr = dr
        self.rho_short, self.rho_long, self.rho_warmup = rho
        self.initParam(hs=hs, dr=dr)

    def initParam(self, hs=256, dr=0.5):
        # gates [kp, ks, kg, gp, gl, qb, ga]
        self.gDict = OrderedDict(
            kp=lambda x: torch.sigmoid(x),  # ponding
            ks=lambda x: torch.sigmoid(x),  # shallow
            kd=lambda x: torch.sigmoid(x),  # deep
            gd=lambda x: torch.sigmoid(x),  # partition of shallow to deep            
            gl=lambda x: torch.pow(torch.sigmoid(x) * 10, 3),  # effective depth
            qb=lambda x: torch.relu(x) / 10,  # baseflow
            ga=lambda x: torch.softmax(x, -1),  # area
            gi=lambda x: F.hardsigmoid(x) / 2,  # interception
            ge=lambda x: torch.relu(x),  # evaporation
        )
        self.kDict = dict(
            km=lambda x: torch.exp(x),  # snow melt
        )
        self.FC = nn.Linear(self.ng, hs)
        self.FC_r = nn.Linear(hs, self.nh * self.nr)
        self.FC_g = nn.Linear(hs, self.nh * len(self.gDict))
        self.FC_kin = nn.Linear(4, hs)
        self.FC_kout = nn.Linear(hs, self.nh * len(self.kDict))

    def getParam(self, x, xc):
        f = x[:, :, 2:]  # T1, T2, Rad and Hum
        nt = x.shape[0]
        state = self.FC(xc)
        mask_k = createMask(state, self.dr)
        mask_g = createMask(state, self.dr)
        mask_r = createMask(state, self.dr)
        pK = self.FC_kout(
            DropMask.apply(torch.tanh(self.FC_kin(f) + state), mask_k, self.training)
        )  # check in debug
        pG = self.FC_g(DropMask.apply(torch.tanh(state), mask_g, self.training))
        pR = self.FC_r(DropMask.apply(torch.tanh(state), mask_r, self.training))
        paramK = sepParam(pK, self.nh, self.kDict)
        paramG = sepParam(pG, self.nh, self.gDict)
        paramR = pR
        return paramK, paramG, paramR

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def initState(self, ns):
        Sf = torch.zeros(ns, self.nh)
        Ss = torch.zeros(ns, self.nh)
        Sg = torch.zeros(ns, self.nh)
        if torch.cuda.is_available():
            Sf = Sf.cuda()
            Ss = Ss.cuda()
            Sg = Sg.cuda()
        return Sf, Ss, Sg

    def forward(self, x, xc, outStep=False):
        nt = x.shape[0]
        ns = x.shape[1]
        Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
        Sf, Ss, Sd = self.initState(ns)
        paramK, paramG, paramR = self.getParam(x, xc)
        Ps, Pl = func.divideP(Prcp, T1, T2)
        Ps = Ps.unsqueeze(-1)
        Pl = Pl.unsqueeze(-1)
        Evp = Evp.unsqueeze(-1)

        def step(k):
            Sf_new, qf = bucket.snow(Sf, Ps[k, ...], paramK['km'][k, ...])
            Is = qf + Pl[k, ...] * paramG['gi'] - Evp[k, ...] * paramG['ge']
            Ss_new, qp, qsA = bucket.shallow(
                Ss, Is, L=paramG['gl'], k1=paramG['kp'], k2=paramG['ks']
            )
            qs = qsA * (1 - paramG['gd'])
            Id = qsA * paramG['gd']
            Sd_new, qd = bucket.deep(Sd, Id, k=paramG['kd'], baseflow=paramG['qb'])
            return (Sf_new, Ss_new, Sd_new), (qp, qs, qd)

        Qp, Qs, Qd = [], [], []
        if outStep:
            Hf, Hs, Hd = [], [], []
        # warmup
        with torch.no_grad():
            for iT in range(self.rho_warmup):
                (Sf, Ss, Sd), (qp, qs, qd) = step(iT)
                for l, v in zip([Qp, Qs, Qd], [qp, qs, qd]):
                    l.append(v)
                if outStep:
                    for l, v in zip([Hf, Hs, Hd], [Sf, Ss, Sd]):
                        l.append(v)
        # forward
        H = [[Ss.detach(), Sd.detach()]]
        for iT in range(self.rho_warmup, nt):
            t0 = time.time()
            if iT < self.rho_warmup + self.rho_short:
                (Sf, Ss, Sd), (qp, qs, qd) = step(iT)
            else:
                Ss, Sd = H[0]
                for i in range(self.rho_short):
                    (Sf, Ss, Sd), (qp, qs, qd) = step(iT)
                _ = H.pop(0)
            if (iT - self.rho_warmup) % self.rho_long == 0:
                Sf.detach_()
            H.append([Ss.detach(), Sd.detach()])
            for l, v in zip([Qp, Qs, Qd], [qp, qs, qd]):
                l.append(v)
            if outStep:
                for l, v in zip([Hf, Hs, Hd], [Sf, Ss, Sd]):
                    l.append(v)
            qOut = torch.sum(
                qp * paramG['ga'] + qs * paramG['ga'] + qd * paramG['ga'], dim=1
            )
        # routing
        QpT = torch.stack(Qp, dim=0)
        QsT = torch.stack(Qs, dim=0)
        QdT = torch.stack(Qd, dim=0)
        QpR = convTS(QpT, paramR) * paramG['ga']
        QsR = convTS(QsT, paramR) * paramG['ga']
        QdR = convTS(QdT, paramR) * paramG['ga']
        # QpR = QpT * paramG['ga']
        # QsR = QsT * paramG['ga']
        # QdR = QdT * paramG['ga']
        qOut = torch.sum(QpR + QsR + QdR, dim=2)
        if outStep:
            Hf, Hs, Hd = [torch.stack(l, dim=0) for l in [Hf, Hs, Hd]]
            return qOut, (QpR, QsR, QdR), (Hf, Hs, Hd)
        else:
            return qOut
