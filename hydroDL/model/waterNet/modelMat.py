import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, func
from hydroDL.model.waterNet.func import convTS, sepParam
from hydroDL.model.dropout import createMask, DropMask
from collections import OrderedDict
import time

# forward with no_grad then record grad


class WaterNet0313(torch.nn.Module):
    def __init__(self, nf, ng, nh, nr, rho=(5, 365, 0), hs=256, dr=0.5):
        super().__init__()
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
            ki=lambda x: torch.relu(x),  # interception
            ke=lambda x: torch.exp(x),  # evaporation
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
        )
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

        # forward all time steps
        SfLst, SsLst, SgLst = [], [], []
        with torch.no_grad():
            for iT in range(nt):
                storage, flux = bucket.step(iT, storage, input, param)
                Sf, Ss, Sd = storage
                SfLst.append(Sf)
                SsLst.append(Ss)
                SgLst.append(Sd)

        # for iT in range(nt):
        #     storage, flux = bucket.step(iT, storage, input, param)
        #     Sf, Ss, Sd = storage
        #     if (iT - self.rho_warmup) % self.rho_long == 0:
        #         Sf=Sf.detach()
        #     SfLst.append(Sf)
        #     SsLst.append(Ss.detach())
        #     SgLst.append(Sd.detach())
        Sf = torch.stack(SfLst, dim=0)
        Ss = torch.stack(SsLst, dim=0)
        Sd = torch.stack(SgLst, dim=0)
        del SfLst, SsLst, SgLst
        if torch.cuda.is_available():
            Sf = Sf.cuda()
            Ss = Ss.cuda()
            Sd = Sd.cuda()
        # short term
