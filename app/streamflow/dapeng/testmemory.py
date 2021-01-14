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
exp_name='parameter_optim'
exp_disp='change_basinnorm'

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

# # train DA model
# nDayLst = [25, 27, 29, 31]
# for nDay in nDayLst:
#     optData = default.update(default.optDataCamels, daObs=nDay)
#     optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
#     optLoss = default.optLoss
#     optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH)
#     save_path = exp_name + '/' + exp_disp + \
#                 '/epochs{}_batch{}_rho{}_hiddensize{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
#                                                               optTrain['miniBatch'][1], optModel['hiddenSize'])
#     out = os.path.join(pathCamels['Out'], save_path, 'All-90-95-DA' + str(nDay))
#     masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
#     # master.runTrain(masterDict, cudaID=cid % 3, screen='test-DA' + str(nDay))
#     cid = cid + 1

# test original model
caseLst = ['All-90-95']
nDayLst = range(1, 100, 2)
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
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'cm'
# # fig = plt.figure(figsize=(8, 5), tight_layout=True)
fig = plot.plotBoxFig(dataBox, keyLst, sharey=False)
fig.patch.set_facecolor('white')
fig.show()
# plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_within30.png", dpi=500)

# plot only NSE box
keyLst=['NSE']
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
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'cm'
# # fig = plt.figure(figsize=(8, 5), tight_layout=True)
xticks = [0] + list(range(3, 100, 4))
xticks = list(map(str, xticks))
fig = plot.plotBoxFig(dataBox, figsize=(10,8), xticklabel=xticks,sharey=False, label1=['DA days'],
                      title='NSE Comparison of DA days')
fig.patch.set_facecolor('white')
fig.show()

# Memory length map
nsedata = list()
for k in range(len(statDictLst)):
    data = statDictLst[k]['NSE']
    nsedata.append(data)
nsedata = np.array(nsedata).T
deltanse = nsedata - np.tile (nsedata[:,0].reshape(-1,1), (1, nsedata.shape[1]))
deltanse = deltanse[:, 1:]
midmemory = list()
fullmemory = list()
nday = range(1, 2*deltanse.shape[1], 2)
for ii in range(deltanse.shape[0]):
    temp = np.where(deltanse[ii, :]<0)[0]
    if temp.size == 0:
        tempmem = 100
    else:
        tempmem = nday[temp[0]] # get the first reduced nse day
    fullmemory.append(tempmem)
    temp = np.where(deltanse[ii, :]<=(deltanse[ii, 0]/2))[0]
    if temp.size == 0:
        tempmidmem = 100
    else:
        tempmidmem = nday[temp[0]]
    midmemory.append(tempmidmem)
# plot memory map
gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
datafull = np.array(fullmemory)
plot.plotMap(datafull, ax=None, lat=gagelat, lon=gagelon, title='Momeory Length', shape=None, cRangeint=True)
datamid = np.array(midmemory)
plot.plotMap(datamid, ax=None, lat=gagelat, lon=gagelon, title='Half Momeory Length', shape=None, cRangeint=True)




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
#
# plot NSE map
gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
data = statDictLst[0]['NSE']
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM', cRange=[0.1, 1.0], shape=None)
gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
data = statDictLst[1]['NSE']
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-1', cRange=[0.1, 1.0], shape=None)
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

# write gages information
gageinfo['LSTMNSE'] = statDictLst[0]['NSE']
gageinfo['NSEDA1'] = statDictLst[1]['NSE']
gageinfo['DeltaNSE'] = deltaNSE
datawrite = pd.DataFrame (gageinfo)
filename = '/mnt/sdb/rnnStreamflow/regional_train/gageinfo.csv'
datawrite.to_csv(filename, index=False,sep=',')
# plot flow regime
