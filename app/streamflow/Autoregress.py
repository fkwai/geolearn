from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels

import numpy as np
import os
from sklearn.linear_model import LinearRegression
import json

# Hyperparameters
nDay=1
savepath = pathCamels['Out'] + '/comparison/Autoreg'
# train default model
optData = default.update(default.optDataCamels, daObs=nDay, rmNan=[True, True])
df, x, y, c = master.master.loadData(optData)
tRange = [19950101, 20000101]
opttestData = default.update(default.optDataCamels, daObs=nDay, tRange=[19950101, 20000101])
dftest, xt, yt, c = master.master.loadData(opttestData)
ngage = x.shape[0]
daylen = xt.shape[1]
Pred = np.full(yt.shape, np.nan)
for ii in range(ngage):
    xdata = x[ii, :,:]
    ydata = y[ii, :,:]
    regmodel = LinearRegression().fit(xdata, ydata)
    xtest = xt[ii, :, :]
    ypred = regmodel.predict(xtest)
    Pred[ii,:,0] = ypred.squeeze()
pred = camels.transNorm(Pred, 'usgsFlow', toNorm=False)
obs = camels.transNorm(yt, 'usgsFlow', toNorm=False)
gageid = 'All'
pred = camels.basinNorm(pred, gageid=gageid, toNorm=False)
obs = camels.basinNorm(obs, gageid=gageid, toNorm=False)
# plot box
statDictLst = [stat.statError(pred.squeeze(), obs.squeeze())]
keyLst=['Bias', 'RMSE', 'NSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(statDictLst)):
        data = statDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)
# plt.style.use('classic')
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["legend.columnspacing"]=0.1
plt.rcParams["legend.handletextpad"]=0.2
labelname = ['LSTM']
nDayLst = [1]
for nDay in nDayLst:
    labelname.append('DA-'+str(nDay))
xlabel = ['Bias ($\mathregular{ft^3}$/s)', 'RMSE ($\mathregular{ft^3}$/s)', 'NSE']
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(10, 5))
fig.patch.set_facecolor('white')
fig.show()

savepath = pathCamels['Out'] + '/comparison/Autoreg'
mFile = os.path.join(savepath, 'evaluation.npy')
if not os.path.isdir(savepath):
    os.makedirs(savepath)
np.save(mFile, statDictLst[0])
# with open(mFile, 'w') as fp:
#     json.dump(statDictLst[0], fp, indent=4)
# print('write master file ' + mFile)
