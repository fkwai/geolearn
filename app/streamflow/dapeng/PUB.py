from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
import pandas as pd

import numpy as np
import os

cid = 0

# Hyperparameters
EPOCH = 200
BATCH_SIZE=10
RHO=365
HIDDENSIZE=256
# Ttrain=[19900101, 19950101]
HUCid = 17

# define directory to save resuts
exp_name='PUB'
# for HUC 11
# exp_disp='HUC'+ str(HUCid)+ '_addexp'
exp_disp='HUC'+ str(HUCid)
# exp_name='parameter_optim'
# exp_disp='change_basinnorm'

# load regional training id
# idfile = pathCamels['Out'] + '/' + exp_name1 + '/' + exp_disp1 + '/badid.txt'
# trainid = np.loadtxt(idfile).astype(int)
# trainid = trainid.tolist()

# read the gages for specific area such as NP and Texas
idfile = '/mnt/sdb/rnnStreamflow/regional_train/gageinfo.csv'
gageframe = pd.read_csv(idfile, sep=',', header=0)
gagese = gageframe[gageframe['huc']==HUCid]
hucid = gagese.id.values.astype(int)
# divide the gages into Train and test
# select 1/10 number of all huc gages as pub gages
pubsize = round(1 / 10 * len(hucid))
pubid = np.random.choice(hucid, size=pubsize, replace=False)
trainid = np.setdiff1d(hucid, pubid)
trainid = trainid.tolist()
pubid = pubid.tolist()

# train default model
optData = default.optDataCamels
optData = default.update(optData, subset=trainid)
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
# nDayLst = [1, 3, 7]
# for nDay in nDayLst:
#     optData = default.update(default.optDataCamels, daObs=nDay)
#     optData = default.update(optData, subset=trainid)
#     optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
#     optLoss = default.optLoss
#     optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH)
#     save_path = exp_name + '/' + exp_disp + \
#                 '/epochs{}_batch{}_rho{}_hiddensize{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
#                                                               optTrain['miniBatch'][1], optModel['hiddenSize'])
#     out = os.path.join(pathCamels['Out'], save_path, 'All-90-95-DA' + str(nDay))
#     masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
#     master.runTrain(masterDict, cudaID=cid % 3, screen='test-DA' + str(nDay))
#     cid = cid + 1

# test original model
caseLst = ['All-90-95']
nDayLst = [1, 3, 7]
for nDay in nDayLst:
    caseLst.append('All-90-95-DA' + str(nDay))
outLst = [os.path.join(pathCamels['Out'], save_path, x) for x in caseLst]
# subset = 'All'
# subset = pubid
tRange = [19950101, 20000101]
predLst = list()
for out in outLst:
    tempdict = master.readMasterFile(out)
    trainid = np.array(tempdict['data']['subset'])
    subset = np.setdiff1d(hucid, trainid).tolist()
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
fig = plot.plotBoxFig(dataBox, keyLst, ['LSTM', 'DA-1', 'DA-3', 'DA-7'], sharey=False, title='Dataset: HUC-'+str(HUCid))
fig.patch.set_facecolor('white')
fig.show()
plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat.png", dpi=500)


# # plot time series
# plt.rcParams['font.size'] = 13
# plt.rcParams['font.family'] = 'Times New Roman'
# t = utils.time.tRange2Array(tRange)
# npub = len(subset)
# fig, axes = plt.subplots(npub, 1, figsize=(16, 14))
# for k in range(npub):
#     # iGrid = np.random.randint(0, 671)
#     # iGrid = np.random.randint(0, len(subset))
#     iGrid = k
#     yPlot = [obs[iGrid, :]]
#     for y in predLst[0:3]:
#         yPlot.append(y[iGrid, :])
#     if k == 0:
#         # plot.plotTS(
#         #     t,
#         #     yPlot,
#         #     ax=axes[k],
#         #     cLst='kbrgm',
#         #     markerLst='-----',
#         #     legLst=['USGS', 'LSTM', 'DA-1', 'DA-3', 'DA-7'])
#         plot.plotTS(
#             t,
#             yPlot,
#             ax=axes[k],
#             cLst='kbrgm',
#             markerLst='-----',
#             legLst=['USGS', 'LSTM: '+str(round(statDictLst[0]['NSE'][k],2)), 'DA-1: '+str(round(statDictLst[1]['NSE'][k],2)),
#                     'DA-3: '+str(round(statDictLst[2]['NSE'][k],2)), 'DA-7:'+str(round(statDictLst[3]['NSE'][k],2))])
#     else:
#         # plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrgm', markerLst='-----')
#         plot.plotTS(
#             t,
#             yPlot,
#             ax=axes[k],
#             cLst='kbrgm',
#             markerLst='-----',
#             legLst=['USGS', 'LSTM: '+str(round(statDictLst[0]['NSE'][k],2)), 'DA-1: '+str(round(statDictLst[1]['NSE'][k],2)),
#                     'DA-3: '+str(round(statDictLst[2]['NSE'][k],2)), 'DA-7: '+str(round(statDictLst[3]['NSE'][k],2))])
# fig.patch.set_facecolor('white')
# fig.show()
# plt.savefig(pathCamels['Out'] + '/' + save_path + "/TS.png", dpi=600)

# plot NSE map
datades = 'HUC_' + str(HUCid)
gageinfo = camels.gageDict
gageidall = gageinfo['id']
C, ind1, ind2 = np.intersect1d(subset, gageidall, return_indices=True)
gagelat = gageinfo['lat'][ind2]
gagelon = gageinfo['lon'][ind2]
# data = statDictLst[0]['NSE']
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM'+'-'+datades, cRange=[0.1, 1.0], shape=None)
# data = statDictLst[1]['NSE']
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-1'+'-'+datades, cRange=[0.1, 1.0], shape=None)
# deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
# plot.plotMap(deltaNSE, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE'+'-'+datades, cRange=[0.4, 1.0], shape=None)
# data = gageframe.LSTMNSE.values
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM'+'-'+'All Dataset', cRange=[0.1, 1.0], shape=None)
# data = gageframe.NSEDA1.values
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-1'+'-'+'All Dataset', cRange=[0.1, 1.0], shape=None)
# data = gageframe.DeltaNSE.values
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE'+'-'+'All Dataset', cRange=[0.4, 1.0], shape=None)

# plot PUB
data = statDictLst[0]['NSE']
C, ind1, ind2 = np.intersect1d(trainid, gageidall, return_indices=True)
trainlat = gageinfo['lat'][ind2]
trainlon = gageinfo['lon'][ind2]
plot.plotPUBloc(data, ax=None, lat=gagelat, lon=gagelon, baclat=trainlat, baclon=trainlon,
                title='HUC-'+str(HUCid)+':LSTM', cRange=[0.0, 1.0])
plt.savefig(pathCamels['Out'] + '/' + save_path + "/LSTM.png", dpi=600)
data = statDictLst[1]['NSE']
plot.plotPUBloc(data, ax=None, lat=gagelat, lon=gagelon, baclat=trainlat, baclon=trainlon,
                title='HUC-'+str(HUCid)+':DA1', cRange=[0.0, 1.0])
plt.savefig(pathCamels['Out'] + '/' + save_path + "/DA1.png", dpi=600)
data = statDictLst[2]['NSE']
plot.plotPUBloc(data, ax=None, lat=gagelat, lon=gagelon, baclat=trainlat, baclon=trainlon,
                title='HUC-'+str(HUCid)+':DA3', cRange=[0.0, 1.0])
plt.savefig(pathCamels['Out'] + '/' + save_path + "/DA3.png", dpi=600)
deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
plot.plotPUBloc(deltaNSE, ax=None, lat=gagelat, lon=gagelon, baclat=trainlat, baclon=trainlon,
                title='HUC-'+str(HUCid)+':Delta NSE', cRange=[0.0, 1.0])
plt.savefig(pathCamels['Out'] + '/' + save_path + "/DeltaNSE.png", dpi=600)

