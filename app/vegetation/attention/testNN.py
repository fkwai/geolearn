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


dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

dm = dbVeg.DataModelVeg(df, subsetName='all')

varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ['Fpar', 'Lai']

indTrain = df.loadSubset('5fold_0_train')
indTest = df.loadSubset('5fold_0_test')


wS = 6
wL = 8
wM = 2

xLst = list()
yLst = list()
for iSite in indTrain:
    [y, t], ind = utils.rmNan([df.y[:, iSite], df.t])
    for k, i in enumerate(ind):
        if i > np.max([wS, wL, wM]) / 2:
            iS = [df.varX.index(var) for var in varS]
            vS = np.nanmean(df.x[i - wS : i + wS, iSite, iS], axis=0)
            iL = [df.varX.index(var) for var in varL]
            vL = np.nanmean(df.x[i - wL : i + wL, iSite, iL], axis=0)
            iM = [df.varX.index(var) for var in varM]
            vM = np.nanmean(df.x[i - wM : i + wM, iSite, iM], axis=0)
            xLst.append(np.concatenate([vS, vL, vM, df.xc[iSite, :]]))
            yLst.append(y[k])
x = np.stack(xLst, axis=0)
y = np.stack(yLst, axis=0)
# drop nan
b1 = np.isnan(x).any(axis=1)
xx = x[~b1, :]
yy = y[~b1]

# data model
mtdXC = ['norm' for i in range(xx.shape[1])]
mtdYC = ['norm']
dm = hydroDL.data.DataModel(XC=xx, YC=yy, mtdXC=mtdXC, mtdYC=mtdYC)
dm.trans()

# build neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

nh = 16
model = nn.Sequential(
    nn.Linear(xx.shape[1], nh),
    nn.ReLU(),
    nn.Linear(nh, nh),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(nh, 1),
)
xT=torch.Tensor(dm.xc)
yT=torch.Tensor(dm.yc)

# loss function
loss_fn = nn.MSELoss(reduction='sum')

# optimizer
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer=optim.Adadelta(model.parameters())

model.train()
for k in range(int(1e4)):
    model.zero_grad(set_to_none=True)
    yP = model(xT)
    loss = loss_fn(yP, yT)
    loss.backward()
    optimizer.step()
    print(k,loss.item())

# test data
xLst = list()
yLst = list()
for iSite in indTest:
    [y, t], ind = utils.rmNan([df.y[:, iSite], df.t])
    for k, i in enumerate(ind):
        if i > np.max([wS, wL, wM]) / 2:
            iS = [df.varX.index(var) for var in varS]
            vS = np.nanmean(df.x[i - wS : i + wS, iSite, iS], axis=0)
            iL = [df.varX.index(var) for var in varL]
            vL = np.nanmean(df.x[i - wL : i + wL, iSite, iL], axis=0)
            iM = [df.varX.index(var) for var in varM]
            vM = np.nanmean(df.x[i - wM : i + wM, iSite, iM], axis=0)
            xLst.append(np.concatenate([vS, vL, vM, df.xc[iSite, :]]))
            yLst.append(y[k])
x = np.stack(xLst, axis=0)
y = np.stack(yLst, axis=0)
# drop nan
b1 = np.isnan(x).any(axis=1)
xx = x[~b1, :]
yy = y[~b1]
dm2 = hydroDL.data.DataModel(XC=xx, YC=yy, mtdXC=mtdXC, mtdYC=mtdYC)
dm2.borrowStat(dm)
xT2=torch.Tensor(dm2.xc)

model.eval()
yT=model(xT)
yP=dm.transOutYC(yT.detach().numpy())
yT2=model(xT2)
yP2=dm2.transOutYC(yT2.detach().numpy())

fig,ax=plt.subplots(2,1)
ax[0].plot(dm.YC,yP,'*')
ax[1].plot(dm2.YC,yP2,'*')
fig.show()

np.corrcoef(dm.YC[:,0],yP[:,0])
np.corrcoef(dm2.YC[:,0],yP2[:,0])