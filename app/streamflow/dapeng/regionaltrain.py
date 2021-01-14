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
BATCH_SIZE=11
RHO=365
HIDDENSIZE=256
# Ttrain=[19900101, 19950101]

# define directory to save resuts
exp_name='regional_train'
exp_disp='badareas_NGP'
# exp_name='parameter_optim'
# exp_disp='change_basinnorm'

# load regional training id
# idfile = pathCamels['Out'] + '/' + exp_name1 + '/' + exp_disp1 + '/badid.txt'
# trainid = np.loadtxt(idfile).astype(int)
# trainid = trainid.tolist()

# read the gages for specific area such as NP and Texas
idfile = '/mnt/sdb/rnnStreamflow/regional_train/Northplainbadlstm.txt'
gageframe = pd.read_csv(idfile, sep=',', header=0)
trainid = gageframe.id.values.astype(int)
trainid = trainid.tolist()

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
subset = trainid
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
fig = plot.plotBoxFig(dataBox, keyLst, ['LSTM', 'DA-1', 'DA-3', 'DA-7'], sharey=False, title='Dataset: North Great Plain')
fig.patch.set_facecolor('white')
fig.show()
# plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_test.png", dpi=500)


# plot time series
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'Times New Roman'
t = utils.time.tRange2Array(tRange)
fig, axes = plt.subplots(5, 1, figsize=(12, 8))
for k in range(5):
    # iGrid = np.random.randint(0, 671)
    iGrid = np.random.randint(0, len(subset))
    yPlot = [obs[iGrid, :]]
    for y in predLst:
        yPlot.append(y[iGrid, :])
    if k == 0:
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst='kbrmg',
            markerLst='-----',
            legLst=['USGS', 'LSTM', 'DA-1', 'DA-3', 'DA-7'])
    else:
        plot.plotTS(t, yPlot, ax=axes[k], cLst='kbrmg', markerLst='-----')
fig.patch.set_facecolor('white')
fig.show()
# plt.savefig(pathCamels['Out'] + '/' + save_path + "/TS.png", dpi=500)

# plot NSE map
datades = 'NGP Bad Gages'
gageinfo = camels.gageDict
gageidall = gageinfo['id']
C, ind1, ind2 = np.intersect1d(subset, gageidall, return_indices=True)
gagelat = gageinfo['lat'][ind2]
gagelon = gageinfo['lon'][ind2]
data = statDictLst[0]['NSE']
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM'+'-'+datades, cRange=[0.0, 1.0], shape=None)
data = statDictLst[1]['NSE']
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-1'+'-'+datades, cRange=[0.0, 1.0], shape=None)
deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
plot.plotMap(deltaNSE, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE'+'-'+datades, cRange=[0.4, 1.0], shape=None)
data = gageframe.LSTMNSE.values
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM'+'-'+'All Dataset', cRange=[0.0, 1.0], shape=None)
data = gageframe.NSEDA1.values
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-1'+'-'+'All Dataset', cRange=[0.0, 1.0], shape=None)
data = gageframe.DeltaNSE.values
plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE'+'-'+'All Dataset', cRange=[0.4, 1.0], shape=None)
# plot regional VS global delta
deltarg = statDictLst[0]['NSE'] - gageframe.LSTMNSE.values
plot.plotMap(deltarg, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE (Bad - All)', cRange=[-0.5, 0.5], shape=None)
# plot box of Delta NSE
plt.rcParams['font.size'] = 16
plt.figure(figsize=(3,5), tight_layout=True)
bp = plt.boxplot(
    deltarg, patch_artist=True, notch=True, showfliers=False)
plt.setp(bp['boxes'], facecolor='r')
plt.xlabel('Delta NSE')
plt.xticks([])
fig.show()


# plot TS map
tsdata=list()
tsdata.append(obs.squeeze())
npred=2
for ii in range(npred):
    tsdata.append(predLst[ii].squeeze())
plot.plotTsMap(dataMap=deltaNSE, dataTs=tsdata, t=t, lat=gagelat, lon=gagelon, tsNameLst=['USGS', 'LSTM', 'DA-1'])

