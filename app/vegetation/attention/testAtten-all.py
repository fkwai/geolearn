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
import torchmetrics


dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

dm = dbVeg.DataModelVeg(df, subsetName='all')

varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ['Fpar', 'Lai']

iS = [df.varX.index(var) for var in varS]
matS = df.x[:, :, iS]
iL = [df.varX.index(var) for var in varL]
matL = df.x[:, :, iL]
iM = [df.varX.index(var) for var in varM]
matM = df.x[:, :, iM]

countS = np.sum(~np.isnan(matS), axis=0)[:, 0]
countL = np.sum(~np.isnan(matL), axis=0)[:, 0]
countM = np.sum(~np.isnan(matM), axis=0)[:, 0]
count = np.sum(~np.isnan(df.y), axis=0)[:, 0]
ind = np.where((countS > 80) & (countL > 80) & (countM > 100) & (count > 15))[0]
nsite = len(ind)
nTrain = math.floor(len(ind) * 0.8)
indTrain = ind[torch.randperm(nsite)[:nTrain]]
indTest = np.setdiff1d(ind, indTrain)

# add date index
dm = DataModel(
    X=df.x[:, ind, :],
    XC=df.xc[
        ind,
    ],
    Y=df.y[:, ind, :],
)
dm1 = DataModel(
    X=df.x[:, indTrain, :],
    XC=df.xc[
        indTrain,
    ],
    Y=df.y[:, indTrain, :],
)
dm2 = DataModel(
    X=df.x[:, indTest, :],
    XC=df.xc[
        indTest,
    ],
    Y=df.y[:, indTest, :],
)

dm.trans(mtdDefault='minmax')
dm1.borrowStat(dm)
dm2.borrowStat(dm)

dataTup1 = dm1.getData()
dataTup2 = dm2.getData()
rho = 45
dataEnd1, (iT1, jT1) = dataTs2Range(dataTup1, rho, returnInd=True)
dataEnd2, (iT2, jT2) = dataTs2Range(dataTup2, rho, returnInd=True)
t1 = df.t[iT1]
t2 = df.t[iT2]

x1, xc1, y1, yc1 = dataEnd1
k = indTrain[0]
bS = 4
bL = 4
bM = 15

# calculate
pSLst, pLLst, pMLst = list(), list(), list()
ns = yc1.shape[0]
nMat = np.zeros([ns, 3])
for k in range(ns):
    tempS = x1[:, k, iS]
    pS = np.where(~np.isnan(tempS).any(axis=1))[0]
    tempL = x1[:, k, iL]
    pL = np.where(~np.isnan(tempL).any(axis=1))[0]
    tempM = x1[:, k, iM]
    pM = np.where(~np.isnan(tempM).any(axis=1))[0]
    pSLst.append(pS)
    pLLst.append(pL)
    pMLst.append(pM)
    nMat[k, :] = [len(pS), len(pL), len(pM)]


# remove all empty
indNan = np.where(nMat == 0)[0]
indKeep = np.setdiff1d(np.arange(ns), indNan)
x1 = x1[:, indKeep, :]
xc1 = xc1[indKeep, :]
yc1 = yc1[indKeep, :]
nMat = nMat[indKeep, :]
pSLst = [pSLst[k] for k in indKeep]
pLLst = [pLLst[k] for k in indKeep]
pMLst = [pMLst[k] for k in indKeep]
ns = yc1.shape[0]


# random sample within window
rS = np.random.randint(0, nMat[:, 0], [bS, ns]).T
rL = np.random.randint(0, nMat[:, 1], [bL, ns]).T
rM = np.random.randint(0, nMat[:, 2], [bM, ns]).T
pS = np.stack([pSLst[k][rS[k, :]] for k in range(ns)], axis=0)
pL = np.stack([pLLst[k][rL[k, :]] for k in range(ns)], axis=0)
pM = np.stack([pMLst[k][rM[k, :]] for k in range(ns)], axis=0)
matS1 = x1[:, :, iS]
matL1 = x1[:, :, iL]
matM1 = x1[:, :, iM]
xS = np.stack([matS1[pS[k, :], k, :] for k in range(ns)], axis=0)
xL = np.stack([matL1[pL[k, :], k, :] for k in range(ns)], axis=0)
xM = np.stack([matM1[pM[k, :], k, :] for k in range(ns)], axis=0)
pS = (pS - rho) / rho
pL = (pL - rho) / rho
pM = (pM - rho) / rho

# test


class InputFeature(nn.Module):
    def __init__(self, nTup, nh):
        super().__init__()
        self.nh = nh
        self.lnLst = nn.ModuleList()
        for n in nTup:
            self.lnLst.append(nn.Sequential(nn.Linear(n, nh, bias=False), nn.Tanh()))
        self.W_o = nn.Linear(nh, nh, bias=False)

    def getPos(self, pos):
        nh = self.nh
        P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
        for i in range(int(nh / 2)):
            P[:, :, 2 * i] = torch.sin(pos * (i + 1) * torch.pi)
            P[:, :, 2 * i + 1] = torch.cos(pos * (i + 1) * torch.pi)
        return P

    def forward(self, xTup, pTup):
        outLst = list()
        for k in range(len(self.lnLst)):
            x = self.lnLst[k](xTup[k]) + self.getPos(pTup[k])
            outLst.append(x)
        out = torch.cat(outLst, dim=1)
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
    def __init__(self, nTup, nh):
        super().__init__()
        self.encoder = InputFeature(nTup, nh)
        self.atten = AttentionLayer(nh, nh)
        self.addnorm1 = AddNorm(nh, 0.1)
        self.addnorm2 = AddNorm(nh, 0.1)
        self.ffn1 = PositionWiseFFN(nh, nh)
        self.ffn2 = PositionWiseFFN(nh, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos):
        xIn = self.encoder(x, pos)
        out = self.atten(xIn)
        out = self.addnorm1(xIn, out)
        out = self.ffn1(out)
        out = self.addnorm2(xIn, out)
        out = self.ffn2(out)
        out = out.squeeze(-1)
        out = out.mean(-1)
        return out


nh = 16
xS = torch.tensor(xS, dtype=torch.float32)
xL = torch.tensor(xL, dtype=torch.float32)
xM = torch.tensor(xM, dtype=torch.float32)
pS = torch.tensor(pS, dtype=torch.float32)
pL = torch.tensor(pL, dtype=torch.float32)
pM = torch.tensor(pM, dtype=torch.float32)
yT = torch.tensor(yc1, dtype=torch.float32)


nTup = (xS.shape[-1], xL.shape[-1], xM.shape[-1])
model = FinalModel(nTup, 16)
yP = model((xS,xL,xM), (pS,pL,pM))
# yP = model((xS, xL), (pS, pL))

loss_fn = nn.L1Loss(reduction='mean')
learning_rate = 1e-2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer=optim.Adadelta(model.parameters())

model.train()
for k in range(int(10000)):
    model.zero_grad()
    yP = model((xS,xL,xM), (pS,pL,pM))
    loss = loss_fn(yP, yT[:, 0])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(k, loss.item())

# test

x2, xc2, y2, yc2 = dataEnd2

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(yP.detach().numpy(), yT, '*')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.show()

np.corrcoef(yP.detach().numpy(), yT[:, 0].detach().numpy())

np.mean(yP.detach().numpy() - yc1) * 350
