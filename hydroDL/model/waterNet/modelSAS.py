import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, func
from hydroDL.model.waterNet.func import convTS, sepParam
from hydroDL.model.dropout import createMask, DropMask
from collections import OrderedDict
import time


class WaterNet0510(torch.nn.Module):
    def __init__(self, nf, ng, nh, rho_warmup=365, hs=256, dr=0.5, nd=366):
        super().__init__()
        self.nf = nf
        self.nh = nh
        self.ng = ng
        # self.nr = nr
        self.nd = nd
        self.hs = hs
        self.dr = dr
        self.rho_warmup = rho_warmup
        self.initParam(hs=hs, dr=dr)

    def initParam(self, hs=256, dr=0.5):
        # gates [kp, ks, kg, gp, gl, qb, ga]
        self.gDict = OrderedDict(
            gk=lambda x: torch.exp(x) / 100,  # curve parameter
            gl=lambda x: torch.exp(x) * 100,  # effective depth
            qb=lambda x: torch.relu(x) / 10,  # baseflow
            ga=lambda x: torch.softmax(x, -1),  # area
            gi=lambda x: F.hardsigmoid(x) / 2,  # interception
            ge=lambda x: torch.relu(x),  # evaporation
        )
        self.kDict = dict(
            km=lambda x: torch.exp(x),  # snow melt
        )
        self.FC = nn.Linear(self.ng, hs)
        # self.FC_r = nn.Linear(hs, self.nh * (self.nr + 1))
        self.FC_g = nn.Linear(hs, self.nh * len(self.gDict))
        self.FC_kin = nn.Linear(4, hs)
        self.FC_kout = nn.Linear(hs, self.nh * len(self.kDict))
        self.reset_parameters()

    def getParam(self, x, xc, raw=False):
        f = x[:, :, 2:]  # T1, T2, Rad and Hum
        nt = x.shape[0]
        state = self.FC(xc)
        mask_k = createMask(state, self.dr)
        mask_g = createMask(state, self.dr)
        # mask_r = createMask(state, self.dr)
        pK = self.FC_kout(
            DropMask.apply(torch.tanh(self.FC_kin(f) + state), mask_k, self.training)
        )
        pG = self.FC_g(DropMask.apply(torch.tanh(state), mask_g, self.training))
        # pR = self.FC_r(DropMask.apply(torch.tanh(state), mask_r, self.training))
        paramK = sepParam(pK, self.nh, self.kDict, raw=raw)
        paramG = sepParam(pG, self.nh, self.gDict, raw=raw)
        # paramR = func.onePeakWeight(pR, self.nh, self.nr)
        # return paramK, paramG, paramR

        return paramK, paramG

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def initState(self, ns):
        Sf = torch.zeros(ns, self.nh)
        D = torch.zeros(ns, self.nh, self.nd)
        if torch.cuda.is_available():
            Sf = Sf.cuda()
            D = D.cuda()
        return Sf, D

    def forwardAll(self, x, xc, outOpt=0):
        # outOpt: 0 for training state initialization
        #         1 for prediction
        nt = x.shape[0]
        ns = x.shape[1]
        Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
        Ps, Pl = func.divideP(Prcp, T1, T2)
        Ps = Ps.unsqueeze(-1)
        Pl = Pl.unsqueeze(-1)
        Evp = Evp.unsqueeze(-1)
        input = [Ps, Pl, Evp]
        # forward all time steps
        SfLst, DLst, fluxLst = [], [], []
        with torch.no_grad():
            storage = self.initState(ns)
            param = self.getParam(x, xc)
            for iT in range(nt):
                storage, flux = bucket.stepSAS(iT, storage, input, param)
                Sf, D = storage
                SfLst.append(Sf)
                DLst.append(D)
                if outOpt == 1:
                    fluxLst.append(flux[1])
        if outOpt == 0:
            return input, param, SfLst, DLst
        elif outOpt == 1:
            return fluxLst, input, param, SfLst, DLst

    def trainModel(
        self, x, xc, y, optim, lossFun, nIterEpoch=5, batchSize=[100, 365], ep=None
    ):
        nbatch, rho = batchSize
        nt = y.shape[0]
        ns = y.shape[1]
        t0 = time.time()
        inputAll, paramAll, SfLst, DLst = self.forwardAll(x, xc)
        print('forward done {}s'.format(time.time() - t0))
        [Ps, Pl, Evp] = inputAll
        # [paramK, paramG, paramR] = paramAll
        [paramK, paramG] = paramAll
        lossEp = 0
        for iIter in range(nIterEpoch):
            t0 = time.time()
            iS = torch.randint(0, ns, [nbatch])
            iT = torch.randint(self.rho_warmup, nt - rho, [nbatch])
            Sf_sub = torch.FloatTensor(nbatch, self.nh)
            D_sub = torch.FloatTensor(nbatch, self.nh, self.nd)
            Ps_sub = torch.FloatTensor(rho, nbatch, 1)
            Pl_sub = torch.FloatTensor(rho, nbatch, 1)
            Evp_sub = torch.FloatTensor(rho, nbatch, 1)
            q_obs = torch.FloatTensor(rho, nbatch)
            x_sub = torch.FloatTensor(rho, nbatch, x.shape[-1])
            xc_sub = torch.FloatTensor(nbatch, xc.shape[-1])

            if torch.cuda.is_available():
                Sf_sub = Sf_sub.cuda()
                D_sub = D_sub.cuda()
                Ps_sub = Ps_sub.cuda()
                Pl_sub = Pl_sub.cuda()
                Evp_sub = Evp_sub.cuda()
                q_obs = q_obs.cuda()
                x_sub = x_sub.cuda()
                xc_sub = xc_sub.cuda()

            # minibatch of input and state
            for k in range(nbatch):
                Sf_sub[k, :] = SfLst[iT[k]][iS[k], :]
                D_sub[k, :, :] = DLst[iT[k]][iS[k], :, :]
                Ps_sub[:, k, :] = Ps[iT[k] : iT[k] + rho, iS[k], :]
                Pl_sub[:, k, :] = Pl[iT[k] : iT[k] + rho, iS[k], :]
                Evp_sub[:, k, :] = Evp[iT[k] : iT[k] + rho, iS[k], :]
                q_obs[:, k] = y[iT[k] : iT[k] + rho, iS[k], 0]
                x_sub[:, k, :] = x[iT[k] : iT[k] + rho, iS[k], :]
                xc_sub[k, :] = xc[iS[k], :]

            param_sub = self.getParam(x_sub, xc_sub)
            input = [Ps_sub, Pl_sub, Evp_sub]
            storage = (Sf_sub, D_sub)
            q_pred = torch.zeros(rho, nbatch)
            if torch.cuda.is_available():
                q_pred = q_pred.cuda()
            for k in range(rho):
                storage, flux = bucket.stepSAS(k, storage, input, param_sub)
                q_bucket = torch.sum(flux[1], dim=-1)
                q_pred[k, :] = torch.sum(q_bucket * param_sub[1]['ga'], dim=-1)
            loss = lossFun(q_pred, q_obs)
            # loss.backward()
            # optim.step()
            if ep is not None:
                print(
                    'ep {} iter {} loss {:.2f} time {:.2f}'.format(
                        ep, iIter, loss.item(), time.time() - t0
                    )
                )
                # # print prameters
                # paramK, paramG = self.getParam(x, xc, raw=True)
                # gk = paramG['gk'].detach().numpy()
                # gl = paramG['gl'].detach().numpy()
                # print('gk', gk)
                # print('gl', gl)
            # print(self.FC._parameters['weight'].grad)
            # print(torch.any(torch.isnan(self.FC._parameters['weight'])))
            loss.backward()
            # print(self.FC._parameters['weight'])
            grad= self.FC._parameters['weight'].grad
            print(torch.max(grad),torch.min(grad))  
            print(torch.max(q_pred),torch.min(q_pred))   
 
            # print(torch.any(torch.isnan(self.FC._parameters['weight'])))
            optim.step()
            lossEp = lossEp + loss.item()
            self.zero_grad()
            optim.zero_grad()
        return lossEp / nIterEpoch
