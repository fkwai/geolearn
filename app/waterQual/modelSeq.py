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

workDir = r'C:\Users\geofk\work\waterQuality'
siteNo = '04086120'
varWqLst = ['00608', '00625', '00631', '00665', '80154']
# varWqLst = ['00060', '00608']
varFcLst = ['ppt', 'tmean']
nTest = 365
rho = 1000
batchSize = 10
nEpoch = 500
nx = len(varFcLst) + 1
ny = len(varWqLst)

# setup result folder
saveFolder = os.path.join(workDir, 'singleSite', siteNo)
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

# load data
dfDaily = usgs.readUsgsText(os.path.join(workDir, 'data', 'dailyTS', siteNo),
                            dataType='dailyTS')
dfForcing = pd.read_csv(os.path.join(workDir, 'data', 'forcing', siteNo))
t = dfDaily['datetime'].values
x = np.ndarray([t.size, len(varFcLst) + 1])
y = np.ndarray([t.size, len(varWqLst)])
for i, var in enumerate(varWqLst):
    y[:, i] = dfDaily[var].values
x[:, 0] = dfDaily['00060'].values
for i, var in enumerate(varFcLst):
    x[:, i + 1] = dfForcing[var].values

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
def subset(x, y, nt, rho):
    iT = np.random.randint(0, nt - rho, [batchSize])
    xTensor = torch.zeros([rho, batchSize, nx], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, ny], requires_grad=False)
    for k in range(batchSize):
        xTensor[:, k, :] = \
            torch.from_numpy(x[np.arange(iT[k], iT[k] +rho), :])
        yTensor[:, k, :] = \
            torch.from_numpy(y[np.arange(iT[k], iT[k] +rho), :])
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


# model
model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=32)
lossFun = crit.RmseLoss()
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())

# training
nt = tTrain.size
if batchSize * rho > nt:
    nIterEp = 1
else:
    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / nt)))
lossEp = 0
t0 = time.time()
model.zero_grad()
model.train()
for iEp in range(1, nEpoch + 1):
    lossEp = 0
    t0 = time.time()
    for iIter in range(nIterEp):
        xT, yT = subset(xTrain, yTrain, nt, rho)
        yP = model(xT)
        loss = lossFun(yP, yT)
        loss.backward()
        optim.step()
        model.zero_grad()
        lossEp = lossEp + loss.item()
    lossEp = lossEp / nIterEp
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, lossEp, ct)
    print(logStr)
modelFile = os.path.join(saveFolder, 'model_Ep' + str(nEpoch) + '.pt')
# torch.save(model, modelFile)

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
dfCode = usgs.readUsgsText(os.path.join(workDir, 'usgs_parameter_code'))
fig, axes = plt.subplots(len(varWqLst), 1)
# axes=[axes]
for i, code in enumerate(varWqLst):
    title = code + ' ' + \
        dfCode['parameter_nm'].loc[dfCode['parameter_cd'] ==code].values[0]
    plot.plotTS(t, [yOut[:, i], y[:, i]],
                ax=axes[i],
                cLst='kr',
                lsLst=['-', '-'],
                mLst=[None, None],
                tBar=t[-1] - np.timedelta64(nTest, 'D'),
                title=title)
    if i + 1 < len(varWqLst): axes[i].set_xticks([])
fig.show()
