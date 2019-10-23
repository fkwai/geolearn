from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels

import numpy as np
import os

cid = 0

# Hyperparameters
EPOCH = 200
BATCH_SIZE=100
RHO=365
HIDDENSIZE=256
# Ttrain=[19900101, 19950101]

# define directory to save results
exp_name='longtermDA'
exp_disp='testprecip'

# train default model
optData = default.optDataCamels
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
optLoss = default.optLoss
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH)
save_path = exp_name + '/' + exp_disp + \
       '/epochs{}_batch{}_rho{}_hiddensize{}'.format(optTrain['nEpoch'],optTrain['miniBatch'][0],
                                                        optTrain['miniBatch'][1],optModel['hiddenSize'])
out = os.path.join(pathCamels['Out'], save_path, 'All-90-95')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
# master.runTrain(masterDict, cudaID=cid % 3, screen='test')
cid = cid + 1

# train DA model
nDayLst = [3, 10, 30, 100, 365]
for nDay in nDayLst:
    optData = default.update(default.optDataCamels, daObs=nDay, damean=True, davar='precipitation')
    optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
    optLoss = default.optLoss
    optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH)
    save_path = exp_name + '/' + exp_disp + \
                '/epochs{}_batch{}_rho{}_hiddensize{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                              optTrain['miniBatch'][1], optModel['hiddenSize'])
    out = os.path.join(pathCamels['Out'], save_path, 'All-90-95-DA' + str(nDay))
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    # master.runTrain(masterDict, cudaID=cid % 3, screen='test-DA' + str(nDay))
    cid = cid + 1

# test original model
caseLst = ['All-90-95']
nDayLst = [3, 10, 30, 100, 365]
for nDay in nDayLst:
    caseLst.append('All-90-95-DA' + str(nDay))
outLst = [os.path.join(pathCamels['Out'], save_path, x) for x in caseLst]
subset = 'All'
tRange = [19950101, 20000101]
predLst = list()
for out in outLst:
    df, pred, obs = master.test(out, tRange=tRange, subset=subset, basinnorm=True, epoch=200)
    # pred=np.maximum(pred,0)
    predLst.append(pred)

# plot box
statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
# keyLst = list(statDictLst[0].keys())
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
for nDay in nDayLst:
    labelname.append('DA-'+str(nDay)+'M')
xlabel = ['Bias ($\mathregular{ft^3}$/s)', 'RMSE ($\mathregular{ft^3}$/s)', 'NSE']
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(10, 5))
fig.patch.set_facecolor('white')
fig.show()
plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_0625.png", dpi=600)


# # plot time series
# plt.rcParams['font.size'] = 13
# plt.rcParams['font.family'] = 'Times New Roman'
# t = utils.time.tRange2Array(tRange)
# fig, axes = plt.subplots(5, 1, figsize=(12, 8))
# for k in range(5):
#     iGrid = np.random.randint(0, 671)
#     yPlot = [obs[iGrid, :]]
#     for y in predLst:
#         yPlot.append(y[iGrid, :])
#     if k == 0:
#         plot.plotTS(
#             t,
#             yPlot,
#             ax=axes[k],
#             cLst='kbrmg',
#             markerLst='-----',
#             legLst=['USGS', 'LSTM', 'DA-1', 'DA-3', 'DA-7'])
#     else:
#         plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrmg', markerLst='-----')
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/TS.png", dpi=500)

# plot NSE map
gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
for ii in range(len(statDictLst)):
    data = statDictLst[ii]['NSE']
    if ii == 0:
        plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM', cRange=[0.1, 1.0], shape=None)
    else:
        plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-Average-'+str(nDayLst[ii-1]),
                     cRange=[0.1, 1.0], shape=None)
deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
plot.plotMap(deltaNSE, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE', shape=None)

# plot TS map
t = utils.time.tRange2Array(tRange)
tsdata=list()
tsdata.append(obs.squeeze())
npred=2
for ii in range(npred):
    tsdata.append(predLst[ii].squeeze())
plot.plotTsMap(dataMap=deltaNSE, dataTs=tsdata, t=t, lat=gagelat, lon=gagelon, tsNameLst=['USGS', 'LSTM', 'DA-1'])

# plot multiple NSE maps
plt.rcParams['font.size'] = 13
gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
fig, axs = plt.subplots(3,2, figsize=(12,8), constrained_layout=True)
axs = axs.flat
for ii in range(len(axs)):
    data = statDictLst[ii]['NSE']
    ax = axs[ii]
    if ii == 0:
        subtitle = 'LSTM'
    else:
        subtitle = 'DA-'+str(nDayLst[ii-1])+'M'
    mm, cs=plot.plotMap(data, ax=ax, lat=gagelat, lon=gagelon, title=subtitle,
                 cRange=[0.0, 1.0], shape=None, clbar=False)
fig.colorbar(cs, ax = axs, shrink=0.7)
# plt.tight_layout()
fig.show()
plt.savefig(pathCamels['Out'] + '/' + save_path + "/NSEMap.png", dpi=600)
