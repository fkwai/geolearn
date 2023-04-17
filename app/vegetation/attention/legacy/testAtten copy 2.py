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

dm.trans(mtdDefault='norm')
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
bS = 8
bL = 8
bM = 15

# calculate
pSLst, pLLst, pMLst = list(), list(), list()
ns = yc1.shape[0]
nMat = np.zeros([ns, 3])
for k in range(ns):
    tempS = x1[:, k, iS]
    pS = np.where(~np.isnan(tempS).all(axis=1))[0]
    tempL = x1[:, k, iL]
    pL = np.where(~np.isnan(tempL).all(axis=1))[0]
    tempM = x1[:, k, iM]
    pM = np.where(~np.isnan(tempM).all(axis=1))[0]
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
    def __init__(self, nx, nh):
        super().__init__()
        self.nh = nh
        self.W = nn.Linear(nx, nh, bias=False)
        self.W_o = nn.Linear(nh, nh, bias=False)

    def forward(self, x):
        x = self.W(x)
        x = self.W_o(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, nx, nh):
        super().__init__()
        self.nh = nh
        self.W_k = nn.Linear(nx, nh, bias=False)
        self.W_q = nn.Linear(nx, nh, bias=False)
        self.W_v = nn.Linear(nx, nh, bias=False)
        self.W_o = nn.Linear(nh, nh, bias=False)

    def getPos(self, pos):
        nh = self.nh
        P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
        for i in range(int(nh / 2)):
            P[:, :, 2 * i] = torch.sin(pos * (i + 1) * torch.pi)
            P[:, :, 2 * i + 1] = torch.cos(pos * (i + 1) * torch.pi)
        return P

    def forward(self, x, pos):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        P = self.getPos(pos)
        q = q + P
        k = k + P
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


xS = torch.tensor(xS, dtype=torch.float32)
pS = torch.tensor(pS, dtype=torch.float32)
nh = 16
input = InputFeature(3, nh)
atten = AttentionLayer(3, nh)
addnorm1 = AddNorm(nh, 0.1)
addnorm2 = AddNorm(nh, 0.1)
ffn1 = PositionWiseFFN(nh, nh)
ffn2 = PositionWiseFFN(nh, 1)

xIn = input(xS)
out = atten(xS, pS)
out = addnorm1(xIn, out)
out = ffn1(out)
out = addnorm2(xIn, out)
out = ffn2(out)
out = out.squeeze(-1)
out = out.sum(-1)


n = 16  # queries
m = 8  # keys pairs
d = 4  # length
v = 2  # value length
nh = 16
inputLayerS = inputLayer(len(varS), nh)
xS = torch.tensor(xS, dtype=torch.float32)
pS = torch.tensor(pS, dtype=torch.float32)

nx = 3
W_k = nn.Linear(nx, nh, bias=False)
W_q = nn.Linear(nx, nh, bias=False)
W_v = nn.Linear(nx, nh, bias=False)
W_o = nn.Linear(nh, nh, bias=False)


q, k, v = W_q(xS), W_k(xS), W_v(xS)
P = getPos(pS, nh)
q = q + P
k = k + P
d = q.shape[1]
score = torch.bmm(q.transpose(1, 2), k) / math.sqrt(d)
attention = torch.softmax(score, dim=-1)
out = torch.bmm(attention, v.transpose(1, 2))
out = W_o(out.transpose(1, 2))

# transformer


pp = PositionWiseFFN(16, 4)
pp(out)

outS = inputLayerS(xS, pS)
inputLayerL = inputLayer(len(varL), nh)
xL = torch.tensor(xL, dtype=torch.float32)
pL = torch.tensor(pL, dtype=torch.float32)
outL = inputLayerL(xL, pL)
inputLayerM = inputLayer(len(varM), nh)
xM = torch.tensor(xM, dtype=torch.float32)
pM = torch.tensor(pM, dtype=torch.float32)
outM = inputLayerM(xM, pM)

xIn = torch.cat([outS, outL, outM], dim=1)


# plot for position
pos = torch.arange(-rho, rho + 1)[None, :] / rho
P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
for i in range(int(nh / 2)):
    P[:, :, 2 * i] = torch.sin(pos * (i + 1) * torch.pi)
    P[:, :, 2 * i + 1] = torch.cos(pos * (i + 1) * torch.pi)
fig, ax = plt.subplots(1, 1)
ax.imshow(P[0, :, :].detach().numpy())
fig.show()


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock(nn.Module):  # @save
    """The Transformer encoder block."""

    def __init__(
        self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """The Transformer encoder."""

    def __init__(
        self,
        num_hiddens,
        ffn_num_hiddens,
        num_heads,
        num_blks,
        dropout,
        use_bias=False,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(
                "block" + str(i),
                TransformerEncoderBlock(
                    num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias
                ),
            )

    def forward(self, X):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


num_hiddens = nh
ffn_num_hiddens = nh
num_heads = 4
num_blks = 2
model = TransformerEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, 0.1)
model(xIn)

test = nn.MultiheadAttention(16, 4, 0.1, batch_first=True)
out = test(xIn, xIn, xIn)
