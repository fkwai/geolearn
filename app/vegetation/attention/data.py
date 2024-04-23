import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import utils
from hydroDL.post import mapplot, axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainBasin
import math
import torch
from torch import nn
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range
import torch.optim as optim
from hydroDL import kPath
import torch.optim.lr_scheduler as lr_scheduler
import dill

rho = 45
dataName = "singleDaily"
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)
dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
siteIdLst = df.siteIdLst
dm.trans(mtdDefault="minmax")
dataTup = dm.getData()
dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
x, xc, y, yc = dataEnd

np.nanmean(dm.x[:, :, 0])
np.nanmax(df.x[:, :, 2])

# calculate position
varS = ["VV", "VH", "vh_vv"]
varL = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ndvi", "ndwi", "nirv"]
varM = ["mod_b{}".format(x) for x in range(1, 8)] + [
    "myd_b{}".format(x) for x in range(1, 8)
]
varF = ["pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr"]

iS = [df.varX.index(var) for var in varS]
iL = [df.varX.index(var) for var in varL]
iM = [df.varX.index(var) for var in varM]
iF = [df.varX.index(var) for var in varF]

pSLst, pLLst, pMLst, pFLst = list(), list(), list(), list()
ns = yc.shape[0]
nMat = np.zeros([yc.shape[0], 4])
for k in range(nMat.shape[0]):
    tempS = x[:, k, iS]
    pS = np.where(~np.isnan(tempS).any(axis=1))[0]
    tempL = x[:, k, iL]
    pL = np.where(~np.isnan(tempL).any(axis=1))[0]
    tempM = x[:, k, iM]
    pM = np.where(~np.isnan(tempM).any(axis=1))[0]
    tempF = x[:, k, iF]
    pF = np.where(~np.isnan(tempF).any(axis=1))[0]
    pSLst.append(pS)
    pLLst.append(pL)
    pMLst.append(pM)
    pFLst.append(pF)
    nMat[k, :] = [len(pS), len(pL), len(pM), len(pF)]

np.where(nMat == 0)
np.sum((np.where(nMat == 0)[1]) == 0)

indKeep = np.where((nMat > 0).all(axis=1))[0]
x = x[:, indKeep, :]
xc = xc[indKeep, :]
yc = yc[indKeep, :]
nMat = nMat[indKeep, :]
pSLst = [pSLst[k] for k in indKeep]
pLLst = [pLLst[k] for k in indKeep]
pMLst = [pMLst[k] for k in indKeep]
pFLst = [pFLst[k] for k in indKeep]
jInd = [jInd[k] for k in indKeep]
siteIdLst = [siteIdLst[k] for k in jInd]

# split train and test
jSite, count = np.unique(jInd, return_counts=True)
countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))])
nRm = sum(countAry[:, 1] < 5)
indSiteAll = countAry[nRm:, 0].astype(int)
dictSubset = dict()
for k in range(5):
    siteTest = indSiteAll[k::5]
    siteTrain = np.setdiff1d(indSiteAll, siteTest)
    indTest = np.where(np.isin(jInd, siteTest))[0]
    indTrain = np.where(np.isin(jInd, siteTrain))[0]
    dictSubset["testSite_k{}5".format(k)] = siteTest.tolist()
    dictSubset["trainSite_k{}5".format(k)] = siteTrain.tolist()
    dictSubset["testInd_k{}5".format(k)] = indTest.tolist()
    dictSubset["trainInd_k{}5".format(k)] = indTrain.tolist()

# save data
# saveFolder = os.path.join(kPath.dirVeg, 'model', 'attention')
# if not os.path.exists(saveFolder):
#     os.mkdir(saveFolder)
# dataFile = os.path.join(saveFolder, 'data.npz')
# np.savez_compressed(dataFile, x=x, xc=xc, y=yc, yc=yc, tInd=iInd, siteInd=jInd)
# subsetFile = os.path.join(saveFolder, 'subset.json')
# with open(subsetFile, 'w') as fp:
#     json.dump(dictSubset, fp, indent=4)


tInd = iInd
siteInd = jInd
trainInd = dictSubset["trainInd_k05"]
testInd = dictSubset["testInd_k05"]

bS = 8
bL = 6
bM = 10
bF = 10


def randomSubset(opt="train", batch=1000):
    # random sample within window
    varS = ["VV", "VH", "vh_vv"]
    varL = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ndvi", "ndwi", "nirv"]
    varM = ["mod_b{}".format(x) for x in range(1, 8)] + [
        "myd_b{}".format(x) for x in range(1, 8)
    ]
    varF = ["pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr"]
    if opt == "train":
        indSel = np.random.permutation(trainInd)[0:batch]
    else:
        indSel = testInd
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    iF = [df.varX.index(var) for var in varF]
    ns = len(indSel)
    rS = np.random.randint(0, nMat[indSel, 0], [bS, ns]).T
    rL = np.random.randint(0, nMat[indSel, 1], [bL, ns]).T
    rM = np.random.randint(0, nMat[indSel, 2], [bM, ns]).T
    rF = np.random.randint(0, nMat[indSel, 3], [bF, ns]).T
    pS = np.stack([pSLst[indSel[k]][rS[k, :]] for k in range(ns)], axis=0)
    pL = np.stack([pLLst[indSel[k]][rL[k, :]] for k in range(ns)], axis=0)
    pM = np.stack([pMLst[indSel[k]][rM[k, :]] for k in range(ns)], axis=0)
    pF = np.stack([pFLst[indSel[k]][rF[k, :]] for k in range(ns)], axis=0)
    matS1 = x[:, indSel, :][:, :, iS]
    matL1 = x[:, indSel, :][:, :, iL]
    matM1 = x[:, indSel, :][:, :, iM]
    matF1 = x[:, indSel, :][:, :, iF]
    xS = np.stack([matS1[pS[k, :], k, :] for k in range(ns)], axis=0)
    xL = np.stack([matL1[pL[k, :], k, :] for k in range(ns)], axis=0)
    xM = np.stack([matM1[pM[k, :], k, :] for k in range(ns)], axis=0)
    xF = np.stack([matF1[pF[k, :], k, :] for k in range(ns)], axis=0)
    pS = (pS - rho) / rho
    pL = (pL - rho) / rho
    pM = (pM - rho) / rho
    pF = (pF - rho) / rho
    xTup = (
        torch.tensor(xS, dtype=torch.float32),
        torch.tensor(xL, dtype=torch.float32),
        torch.tensor(xM, dtype=torch.float32),
        torch.tensor(xF, dtype=torch.float32),
    )
    pTup = (
        torch.tensor(pS, dtype=torch.float32),
        torch.tensor(pL, dtype=torch.float32),
        torch.tensor(pM, dtype=torch.float32),
        torch.tensor(pF, dtype=torch.float32),
    )
    xcOut = torch.tensor(xc[indSel, :], dtype=torch.float32)
    ycOut = torch.tensor(yc[indSel, 0], dtype=torch.float32)
    return xTup, pTup, xcOut, ycOut


class InputFeature(nn.Module):
    def __init__(self, nTup, nxc, nh):
        super().__init__()
        self.nh = nh
        self.lnXc = nn.Sequential(nn.Linear(nxc, nh), nn.ReLU(), nn.Linear(nh, nh))
        self.lnLst = nn.ModuleList()
        for n in nTup:
            self.lnLst.append(
                nn.Sequential(nn.Linear(n, nh), nn.ReLU(), nn.Linear(nh, nh))
            )

    def getPos(self, pos):
        nh = self.nh
        P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
        for i in range(int(nh / 2)):
            P[:, :, 2 * i] = torch.sin(pos / (i + 1) * torch.pi)
            P[:, :, 2 * i + 1] = torch.cos(pos / (i + 1) * torch.pi)
        return P

    def forward(self, xTup, pTup, xc):
        outLst = list()
        for k in range(len(xTup)):
            x = self.lnLst[k](xTup[k]) + self.getPos(pTup[k])
            outLst.append(x)
        outC = self.lnXc(xc)
        out = torch.cat(outLst + [outC[:, None, :]], dim=1)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, nx, nh):
        super().__init__()
        self.nh = nh
        self.W_k = nn.Linear(nx, nh, bias=False)
        self.W_q = nn.Linear(nx, nh, bias=False)
        self.W_v = nn.Linear(nx, nh, bias=False)
        self.W_o = nn.Linear(nh, nh, bias=False)

    def forward(self, x):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        d = q.shape[1]
        score = torch.bmm(q.transpose(1, 2), k) / math.sqrt(d)
        attention = torch.softmax(score, dim=-1)
        out = torch.bmm(attention, v.transpose(1, 2))
        out = self.W_o(out.transpose(1, 2))
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, nh, ny):
        super().__init__()
        self.dense1 = nn.Linear(nh, nh)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(nh, ny)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class FinalModel(nn.Module):
    def __init__(self, nTup, nxc, nh):
        super().__init__()
        self.nTup = nTup
        self.nxc = nxc
        self.encoder = InputFeature(nTup, nxc, nh)
        self.atten = AttentionLayer(nh, nh)
        self.addnorm1 = AddNorm(nh, 0.1)
        self.addnorm2 = AddNorm(nh, 0.1)
        self.ffn1 = PositionWiseFFN(nh, nh)
        self.ffn2 = PositionWiseFFN(nh, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos, xcT, lTup):
        xIn = self.encoder(x, pos, xcT)
        out = self.atten(xIn)
        out = self.addnorm1(xIn, out)
        out = self.ffn1(out)
        out = self.addnorm2(xIn, out)
        out = self.ffn2(out)
        out = out.squeeze(-1)
        k = 0
        temp = 0
        for i in lTup:
            temp = temp + out[:, k : i + k].mean(-1)
            k = k + i
        temp = temp + out[:, k:].mean(-1)
        return temp


nh = 32
xTup, pTup, xcT, yT = randomSubset()

nTup = [x.shape[-1] for x in xTup]
lTup = [x.shape[1] for x in xTup]

# nTup = (xS.shape[-1], xL.shape[-1])
nxc = xc.shape[-1]
model = FinalModel(nTup, nxc, nh)
yP = model(xTup, pTup, xcT, lTup)
# yP = model((xS, xL), (pS, pL))

loss_fn = nn.L1Loss(reduction="mean")
learning_rate = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adadelta(model.parameters())
# scheduler = lr_scheduler.LinearLR(
#     optimizer, start_factor=1.0, end_factor=0.01, total_iters=500
# )


import time

model.train()
nEp = 500
nIterEp = 20
for ep in range(nEp):
    lossEp = 0
    for i in range(nIterEp):
        t0 = time.time()
        xTup, pTup, xcT, yT = randomSubset()
        yP = model(xTup, pTup, xcT, lTup)
        # xS, xL, xM, pS, pL, pM, xcT, yT = randomSubset()
        t1 = time.time()
        model.zero_grad()
        # yP = model((xS, xL, xM), (pS, pL, pM), xcT, lTup)
        yP = model(xTup, pTup, xcT, lTup)
        loss = loss_fn(yP, yT)
        loss.backward()
        t2 = time.time()
        lossEp = lossEp + loss.item()
        optimizer.step()
    optimizer.zero_grad()
    xTup, pTup, xcT, yT = randomSubset()
    yP = model(xTup, pTup, xcT, lTup)
    loss = loss_fn(yP, yT)
    corr = np.corrcoef(yP.detach().numpy(), yT.detach().numpy())[0, 1]
    # if ep > 200:
    #     scheduler.step()
    print(
        "{} {:.3f} {:.3f} {:.3f} time {:.2f} {:.2f}".format(
            ep, lossEp / nIterEp, loss.item(), corr, t1 - t0, t2 - t1
        )
    )

# save results
saveFolder = os.path.join(kPath.dirVeg, "model", "attention")
torch.save(model.state_dict(), os.path.join(saveFolder, "model"))
# json save subset dict
with open(os.path.join(saveFolder, "subset.json"), "w") as fp:
    json.dump(dictSubset, fp, indent=4)


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(yP.detach().numpy(), yT, "*")
fig.show()
np.corrcoef(yP.detach().numpy(), yT.detach().numpy())

# test
saveFolder = os.path.join(kPath.dirVeg, "model", "attention")
model.load_state_dict(torch.load(os.path.join(saveFolder, "model")))
model.eval()
varS = ["VV", "VH", "vh_vv"]
varL = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ndvi", "ndwi", "nirv"]
varM = ["Fpar", "Lai"]
iS = [df.varX.index(var) for var in varS]
iL = [df.varX.index(var) for var in varL]
iM = [df.varX.index(var) for var in varM]
yOut = np.zeros(len(testInd))

for k, ind in enumerate(testInd):
    xS = x[pSLst[ind], ind, :][:, iS][None, ...]
    xL = x[pLLst[ind], ind, :][:, iL][None, ...]
    xM = x[pMLst[ind], ind, :][:, iM][None, ...]
    pS = (pSLst[ind][None, ...] - rho) / rho
    pL = (pLLst[ind][None, ...] - rho) / rho
    pM = (pMLst[ind][None, ...] - rho) / rho
    xcT = xc[ind][None, ...]
    xS = torch.from_numpy(xS).float()
    xL = torch.from_numpy(xL).float()
    xM = torch.from_numpy(xM).float()
    pS = torch.from_numpy(pS).float()
    pL = torch.from_numpy(pL).float()
    pM = torch.from_numpy(pM).float()
    xcT = torch.from_numpy(xcT).float()
    lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
    yP = model((xS, xL, xM), (pS, pL, pM), xcT, lTup)
    yOut[k] = yP.detach().numpy()

yT = yc[testInd, 0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(yOut, yT, "*")
fig.show()

obs = dm.transOutY(yT[:, None])[:, 0]
pred = dm.transOutY(yOut[:, None])[:, 0]
fig, ax = plt.subplots(1, 1)
ax.plot(pred, obs, "*")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
vmin = np.min([xlim[0], ylim[0]])
vmax = np.max([xlim[1], ylim[1]])
_ = ax.plot([vmin, vmax], [vmin, vmax], "r-")
fig.show()

# rmse
np.sqrt(np.mean((obs - pred) ** 2))
np.corrcoef(yOut, yT)
np.mean(pred) - np.mean(obs)


# # save work space
# saveFolder = os.path.join(kPath.dirVeg, "model", "attention")
# # dill.dump_session(os.path.join(saveFolder, 'workspace.db'))
# dill.load_session(os.path.join(saveFolder, "workspace.db"))

# plots
# import matplotlib

# xS = torch.ones(1, 5, 3)
# xL = torch.ones(1, 5, 8)
# xM = torch.ones(1, 5, 2)
# pS = torch.tensor([-1, -0.5, 0, 0.5, 1])[None, :]
# pL = torch.tensor([-1, -0.5, 0, 0.5, 1])[None, :]
# pM = torch.tensor([-1, -0.5, 0, 0.5, 1])[None, :]
# xcT = torch.ones(1, 15)
# yP = model((xS, xL, xM), (pS, pL, pM), xcT, (5, 5, 5))
# xIn = model.encoder((xS, xL, xM), (pS, pL, pM), xcT)
# atten = model.atten
# q, k, v = atten.W_q(xIn), atten.W_k(xIn), atten.W_v(xIn)
# d = q.shape[1]
# score = torch.bmm(q.transpose(1, 2), k) / math.sqrt(d)
# fig, axes = plt.subplots(2, 1)
# im1 = axes[0].imshow(k[0, :, :].detach().numpy())
# im2 = axes[1].imshow(k[0, :, :].detach().numpy())
# fig.colorbar(im1)
# fig.colorbar(im2)
# fig.show()


# matplotlib.rcParams.update({"font.size": 11})
# matplotlib.rcParams.update({"lines.linewidth": 2})
# matplotlib.rcParams.update({"lines.markersize": 12})
# matplotlib.rcParams.update({"legend.fontsize": 11})

# # attention layer
# model.eval()


# obs = dm.transOutY(yT[:, None])[:, 0]
# pred = dm.transOutY(yOut[:, None])[:, 0]
# fig, ax = plt.subplots(1, 1)
# ax.plot(pred, obs, ".")
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# vmin = np.min([xlim[0], ylim[0]])
# vmax = np.max([xlim[1], ylim[1]])
# _ = ax.plot([vmin, vmax], [vmin, vmax], "r-")
# fig.show()

# # to site
# tempS = jInd[testInd]
# tempT = iInd[testInd]
# testSite = np.unique(tempS)
# siteLst = list()
# matResult = np.ndarray([len(testSite), 3])
# for i, k in enumerate(testSite):
#     ind = np.where(tempS == k)[0]
#     t = df.t[tempT[ind]]
#     siteName = df.siteIdLst[k]
#     siteLst.append([pred[ind], obs[ind], t])
#     matResult[i, 0] = np.mean(pred[ind])
#     matResult[i, 1] = np.mean(obs[ind])
#     matResult[i, 2] = np.corrcoef(pred[ind], obs[ind])[0, 1]


# # mean
# fig, ax = plt.subplots(1, 1)
# ax.plot(matResult[:, 0], matResult[:, 1], "*")
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# vmin = np.min([xlim[0], ylim[0]])
# vmax = np.max([xlim[1], ylim[1]])
# _ = ax.plot([vmin, vmax], [vmin, vmax], "r-")
# fig.show()
# np.corrcoef(matResult[:, 0], matResult[:, 1])[0, 1]
# # rmse
# np.sqrt(np.mean((matResult[:, 0] - matResult[:, 1]) ** 2))

# # anomoly
# fig, ax = plt.subplots(1, 1)
# aLst, bLst = list(), list()
# for site in siteLst:
#     aLst.append(site[:, 0] - np.mean(site[:, 0]))
#     bLst.append(site[:, 1] - np.mean(site[:, 1]))
# a, b = np.concatenate(aLst), np.concatenate(bLst)
# ax.plot(np.concatenate(aLst), np.concatenate(bLst), ".")
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# vmin = np.min([xlim[0], ylim[0]])
# vmax = np.max([xlim[1], ylim[1]])
# _ = ax.plot([vmin, vmax], [vmin, vmax], "r-")
# fig.show()

# # mean
# ax.plot(matResult[:, 0], matResult[:, 1], "*")
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# vmin = np.min([xlim[0], ylim[0]])
# vmax = np.max([xlim[1], ylim[1]])
# _ = ax.plot([vmin, vmax], [vmin, vmax], "r-")
# fig.show()

# # correlation map
# import matplotlib.gridspec as gridspec

# trainSite = np.unique(jInd[trainInd])
# lat = df.lat[trainSite]
# lon = df.lon[trainSite]
# figM = plt.figure(figsize=(8, 6))
# gsM = gridspec.GridSpec(1, 1)
# axM = mapplot.mapPoint(
#     figM, gsM[0, 0], lat, lon, np.zeros(len(lat)), cmap="gray", cb=False
# )
# figM.show()

# lat = df.lat[testSite]
# lon = df.lon[testSite]
# figM = plt.figure(figsize=(8, 6))
# gsM = gridspec.GridSpec(1, 1)
# axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, matResult[:, 2], s=50)
# figM.show()


# def funcM():
#     lat = df.lat[testSite]
#     lon = df.lon[testSite]
#     figM = plt.figure(figsize=(8, 6))
#     gsM = gridspec.GridSpec(1, 1)
#     axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, matResult[:, 2], s=50)
#     figP, axP = plt.subplots(1, 1)
#     return figM, axM, figP, axP, lon, lat


# def funcP(iP, axP):
#     print(iP)
#     axP.plot(siteLst[iP][2], siteLst[iP][0], "r*-", label="pred")
#     axP.plot(siteLst[iP][2], siteLst[iP][1], "b*-", label="obs")
#     axP.legend()


# figplot.clickMap(funcM, funcP)

# # position encoding plot
# pos = torch.arange(-45, 45, dtype=torch.float32) / 45
# pos = pos[None, :]
# nh = 32
# P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
# for i in range(int(nh / 2)):
#     # P[:, :, 2 * i] = torch.sin(pos * (i + 1) * torch.pi)
#     # P[:, :, 2 * i + 1] = torch.cos(pos * (i + 1) * torch.pi)
#     P[:, :, 2 * i] = torch.sin(pos * rho / 10000 ** ((i + 1) / 32))
#     P[:, :, 2 * i + 1] = torch.cos(pos * rho / 10000 ** ((i + 1) / 32))
# fig, ax = plt.subplots(1, 1)
# ax.imshow(P[0, :, :].detach().numpy(), extent=[0, 32, -45, 45])

# fig.show()
