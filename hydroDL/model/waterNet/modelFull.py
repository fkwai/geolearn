import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, func
from hydroDL.model.waterNet.func import convTS, sepParam
from hydroDL.model.dropout import createMask, DropMask
from collections import OrderedDict
import time
from sys import getsizeof


def defineFCNN(nx, ny, hs=256, dr=0.5):
    model = nn.Sequential(
        nn.Linear(nx, hs), nn.Tanh(), nn.Dropout(p=dr), nn.Linear(hs, ny)
    )
    return model


class WaterNet0313(torch.nn.Module):
    def __init__(self, nf, ng, nh, nr, rho=(5, 365, 0), hs=256, dr=0.5):
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
        self.FC_r = nn.Linear(hs, self.nh * (self.nr + 1))
        self.FC_g = nn.Linear(hs, self.nh * len(self.gDict))
        self.FC_kin = nn.Linear(4, hs)
        self.FC_kout = nn.Linear(hs, self.nh * len(self.kDict))
        self.reset_parameters()

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
        paramR = func.onePeakWeight(pR, self.nh, self.nr)
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
        storage = self.initState(ns)
        paramK, paramG, paramR = self.getParam(x, xc)
        Ps, Pl = func.divideP(Prcp, T1, T2)
        Ps = Ps.unsqueeze(-1)
        Pl = Pl.unsqueeze(-1)
        Evp = Evp.unsqueeze(-1)
        input = [Ps, Pl, Evp]
        param = [paramK, paramG, paramR]
        if outStep:
            Hf, Hs, Hd = [], [], []
            Of, Op, Os, Od = [], [], [], []

        def saveStep(b):
            if b:
                Sf, Ss, Sd = storage
                qf, qp, qs, qd = flux
                (qf, qp, qs, qd)
                for l, v in zip([Hf, Hs, Hd], [Sf, Ss, Sd]):
                    l.append(v)
                for l, v in zip([Of, Op, Os, Od], [qf, qp, qs, qd]):
                    l.append(v)

        # warmup
        with torch.no_grad():
            for iT in range(self.rho_warmup):
                storage, flux = bucket.step(iT, storage, input, param)
                saveStep(outStep)
        # forward
        H = []
        Q = []
        qOut = torch.zeros(nt - self.nr - self.rho_warmup + 1, ns)
        if torch.cuda.is_available():
            qOut = qOut.cuda()
        for iT in range(self.rho_warmup, nt):
            t0 = time.time()
            # print(iT,storage[1].sum(),'before')
            if iT < self.rho_warmup + self.rho_short:
                storage, flux = bucket.step(iT, storage, input, param)
            else:                
                storage = H[0]
                for i in range(iT - self.rho_short + 1, iT + 1):
                    # print(i,storage[1].sum(),'re-before')
                    storage, flux = bucket.step(i, storage, input, param)
                    # print(i,storage[1].sum(),'re-after')
                _ = H.pop(0)
            # print(iT,storage[1].sum(),'after')
            Sf, Ss, Sd = storage
            qf, qp, qs, qd = flux
            if (iT - self.rho_warmup) % self.rho_long == 0:
                Sf.detach_()
            H.append([Sf, Ss.detach(), Sd.detach()])
            # routing
            Q.append([qp, qs, qd])
            if iT >= self.rho_warmup + self.nr - 1:
                Qp = torch.stack([Q[x][0] for x in range(self.nr)], dim=0)
                Qs = torch.stack([Q[x][1] for x in range(self.nr)], dim=0)
                Qd = torch.stack([Q[x][2] for x in range(self.nr)], dim=0)
                QpR = torch.sum(Qp * paramR, dim=0)
                QsR = torch.sum(Qs * paramR, dim=0)
                QdR = torch.sum(Qd * paramR, dim=0)
                out = torch.sum(
                    QpR * paramG['ga'] + QsR * paramG['ga'] + QdR * paramG['ga'], dim=1
                )
                qOut[iT - self.rho_warmup - self.nr + 1, :] = out
                Q.pop(0)
            saveStep(outStep)
        if outStep:
            Hf, Hs, Hd = [torch.stack(l, dim=0) for l in [Hf, Hs, Hd]]
            Of, Op, Os, Od = [torch.stack(l, dim=0) for l in [Of, Op, Os, Od]]
            return qOut, (Of, Op, Os, Od), (Hf, Hs, Hd)
        else:
            return qOut
