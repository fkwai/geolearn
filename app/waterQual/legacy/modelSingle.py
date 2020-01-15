import os
import numpy as np
import pandas as pd
import torch
import json
import time
import matplotlib.pyplot as plt
from hydroDL.data import usgs
from hydroDL.model import rnn, crit
from hydroDL.post import plot
from hydroDL import utils

workDir = r'C:\Users\geofk\work\waterQuality'
siteNo = '04086120'
varWqLst = ['00608', '00625', '00631', '00665', '80154']
# varWqLst = ['00060', '00608']
varFcLst = ['ppt', 'tmean']
nTest = 365
rho = 365
batchSize = 50
nEpoch = 100
hiddenSize = 64
nx = len(varFcLst) + 1
ny = len(varWqLst)

# setup result folder
saveFolder = os.path.join(workDir, 'singleSite', siteNo)
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

# load sample data
dfSample = usgs.readUsgsText(os.path.join(workDir, 'data', 'sample', siteNo),
                             dataType='sample')
tS = dfSample['datetime'].values
tY = np.arange(tS[0].astype('datetime64[D]'),
               tS[-1].astype('datetime64[D]') + np.timedelta64(1, 'D'),
               np.timedelta64(1, 'D'))
dataY = np.full([tY.size, len(varWqLst)], np.nan)
for i, d in enumerate(tY):
    ind = np.where(tS.astype('datetime64[D]') == d)[0]
    for j, var in enumerate(varWqLst):
        dataY[i, j] = np.nanmean(dfSample[var].values[ind])

# load forcing and streamflow data
dfDaily = usgs.readUsgsText(os.path.join(workDir, 'data', 'dailyTS', siteNo),
                            dataType='dailyTS')
dfForcing = pd.read_csv(os.path.join(workDir, 'data', 'forcing', siteNo))
tX = dfDaily['datetime'].values
dataX = np.ndarray([tX.size, len(varFcLst) + 1])
dataX[:, 0] = dfDaily['00060'].values
for i, var in enumerate(varFcLst):
    dataX[:, i + 1] = dfForcing[var].values

# summarize them
sd = np.max([tX[0], tY[0] - np.timedelta64(rho, 'D')]).astype('datetime64[D]')
ed = np.min([tX[-1], tY[-1]]).astype('datetime64[D]')
t = np.arange(sd, ed + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
y = np.full([t.size, len(varWqLst)], np.nan)
ind1, ind2 = utils.time.intersect(t, tY)
y[ind1, :] = dataY[ind2, :]
y[:rho, :] = np.nan
x = np.full([t.size, len(varFcLst) + 1], np.nan)
ind1, ind2 = utils.time.intersect(t, tX)
x[ind1, :] = dataX[ind2, :]

# normalize
statDict = dict(xMean=np.nanmean(x, axis=0).tolist(),
                xStd=np.nanstd(x, axis=0).tolist(),
                yMean=np.nanmean(y, axis=0).tolist(),
                yStd=np.nanstd(y, axis=0).tolist())
with open(os.path.join(saveFolder, 'stat.json'), 'w') as fp:
    json.dump(statDict, fp)
xNorm = (x - np.tile(statDict['xMean'], [t.size, 1])) / np.tile(
    statDict['xStd'], [t.size, 1])
yNorm = (y - np.tile(statDict['yMean'], [t.size, 1])) / np.tile(
    statDict['yStd'], [t.size, 1])

# seperate train / test
tTrain = t[:-nTest]
tTest = t[-nTest:]
xTrain = xNorm[:-nTest, :]
xTest = xNorm[-nTest:, :]
yTrain = yNorm[:-nTest, :]
yTest = yNorm[-nTest:, :]


# random subset
def subset(x, y, tY, rho):
    iT = indY[np.random.randint(0, tY.size, [batchSize])]
    xTensor = torch.zeros([rho, batchSize, nx], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, ny], requires_grad=False)
    for k in range(batchSize):
        xTensor[:, k, :] = \
            torch.from_numpy(x[np.arange(iT[k]-rho+1, iT[k]+1), :])
        yTensor[:, k, :] = \
            torch.from_numpy(y[np.arange(iT[k]-rho+1, iT[k]+1), :])
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


# model
model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=hiddenSize)
lossFun = crit.RmseLoss()
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())

# training
tY = np.where(~np.isnan(np.sum(yTrain, axis=1)))[0]
if batchSize > tY.size:
    nIterEp = 1
else:
    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize / tY.size)))
lossEp = 0
lossEpLst = list()
t0 = time.time()
model.zero_grad()
model.train()
for iEp in range(1, nEpoch + 1):
    lossEp = 0
    t0 = time.time()
    for iIter in range(nIterEp):
        xT, yT = subset(xTrain, yTrain, tY, rho)
        yP = model(xT)
        loss = lossFun(yP[-1:,:,:], yT[-1:,:,:])
        loss.backward()
        optim.step()
        model.zero_grad()
        lossEp = lossEp + loss.item()
    lossEp = lossEp / nIterEp
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, lossEp, ct)
    print(logStr)
    lossEpLst.append(lossEp)
modelFile = os.path.join(saveFolder, 'model_Ep' + str(nEpoch) + '.pt')
torch.save(model, modelFile)

# predict
# modelFile = os.path.join(saveFolder, 'model_Ep' + str(nEpoch) + '.pt')
# model = torch.load(modelFile)
xT = torch.zeros([xNorm.shape[0], 1, xNorm.shape[1]], requires_grad=False)
xT[:, 0, :] = torch.from_numpy(xNorm)
if torch.cuda.is_available():
    xT = xT.cuda()
    model = model.cuda()
yT = model(xT)
yOut = yT.detach().cpu().numpy().squeeze()
yOut = yOut * np.tile(statDict['yStd'], [t.size, 1]) +\
     np.tile(statDict['yMean'], [t.size, 1])

# plot
# training loss
fig, ax = plt.subplots(1, 1)
plt.plot(np.arange(nEpoch), lossEpLst)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
fig.show()
# forcing
fig, axes = plt.subplots(len(varFcLst) + 1, 1)
titleLst = ['discharge', 'precipitation', 'temperature']
cLst = 'gbr'
for i, title in enumerate(titleLst):
    plot.plotTS(t,
                x[:, i],
                ax=axes[i],
                cLst=cLst[i],
                lsLst=['-'],
                tBar=t[-1] - np.timedelta64(nTest, 'D'),
                title=title)
    if i + 1 < len(varWqLst): axes[i].set_xticks([])
fig.show()
# water quality
dfCode = usgs.readUsgsText(os.path.join(workDir, 'usgs_parameter_code'))
fig, axes = plt.subplots(len(varWqLst), 1)
for i, code in enumerate(varWqLst):
    title = code + ' ' + \
        dfCode['parameter_nm'].loc[dfCode['parameter_cd'] ==code].values[0]
    plot.plotTS(t, [y[:, i], yOut[:, i]],
                ax=axes[i],
                cLst='kr',
                lsLst=['-', '-'],
                mLst=['*', None],
                tBar=t[-1] - np.timedelta64(nTest, 'D'),
                title=title)
    if i + 1 < len(varWqLst): axes[i].set_xticks([])
fig.show()
