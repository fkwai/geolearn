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
Ttrain=[19991001, 20081001]

# define directory to save results
exp_name='parameter_optim'
exp_disp='change_basinnorm/change_loss/MSE'

# train default model
optData = default.optDataCamels
optData = default.update(optData, tRange=Ttrain)
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
optLoss = default.optLossMSE
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH)
# save_path = exp_name + '/' + exp_disp + \
#        '/epochs{}_batch{}_rho{}_hiddensize{}'.format(optTrain['nEpoch'],optTrain['miniBatch'][0],
#                                                         optTrain['miniBatch'][1],optModel['hiddenSize'])
save_path = exp_name + '/' + exp_disp + \
       '/epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'],optTrain['miniBatch'][0],
                                                        optTrain['miniBatch'][1],optModel['hiddenSize'],
                                                        optData['tRange'][0], optData['tRange'][1])
out = os.path.join(pathCamels['Out'], save_path, 'All-90-95')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
# master.runTrain(masterDict, cudaID=cid % 3, screen='test')
cid = cid + 1

# # train DA model
# nDayLst = [1,3]
# for nDay in nDayLst:
#     optData = default.update(default.optDataCamels, daObs=nDay)
#     optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
#     optLoss = default.optLossNSE
#     optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH)
#     out = os.path.join(pathCamels['Out'], save_path, 'All-90-95-DA' + str(nDay))
#     masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
#     master.runTrain(masterDict, cudaID=cid % 3, screen='test-DA' + str(nDay))
#     cid = cid + 1


# test original model
caseLst = ['All-90-95']
nDayLst = [1,3]
for nDay in nDayLst:
    caseLst.append('All-90-95-DA' + str(nDay))
outLst = [os.path.join(pathCamels['Out'], save_path, x) for x in caseLst]
subset = 'All'
tRange = [19891001, 19991001]
predLst = list()
for out in outLst:
    df, pred, obs = master.test(out, tRange=tRange, subset=subset, basinnorm=True, epoch=200)
    # pred=np.maximum(pred,0)
    ## change the units ft3/s to m3/s
    obs = obs*0.0283168
    pred = pred*0.0283168
    predLst.append(pred)

# change the units into cms

# plot box
statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
# keyLst = list(statDictLst[0].keys())
keyLst=['Bias', 'NSE', 'FLV', 'FHV']
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
    labelname.append('DI('+str(nDay)+')')
xlabel = ['Bias ($\mathregular{m^3}$/s)', 'NSE', 'FLV(%)', 'FHV(%)']
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(12, 5))
fig.patch.set_facecolor('white')
fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_0716.png", dpi=600)

# write evaluation results
mFile = os.path.join(pathCamels['Out'], save_path, 'evaluation.npy')
np.save(mFile, statDictLst)

obsFile = os.path.join(pathCamels['Out'], save_path, 'obs.npy')
np.save(obsFile, obs)

predFile = os.path.join(pathCamels['Out'], save_path, 'pred.npy')
np.save(predFile, predLst)

# # plot the comparison of LSTMDA with Auto and ANN
# fname_ANN = pathCamels['Out'] + '/comparison/ANN/hid1_256hid2_256batch_36500epoch_200/evaluation.npy'
# stadic_ann = np.load(fname_ANN, allow_pickle=True).item()
# fname_Auto = pathCamels['Out'] + '/comparison/Autoreg/evaluation.npy'
# stadic_auto = np.load(fname_Auto, allow_pickle=True).item()
# keyLst=['Bias', 'NSE']
# dataBox = list()
# plotDictLst = [statDictLst[0], statDictLst[1], stadic_auto, stadic_ann]
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(plotDictLst)):
#         data = plotDictLst[k][statStr]
#         data = data[~np.isnan(data)]
#         temp.append(data)
#     dataBox.append(temp)
# labelname = ['LSTM', 'DI(1)', 'ARp(1)', 'ANN(1)']
# xlabel = ['Bias ($\mathregular{ft^3}$/s)', 'NSE']
# fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 4))
# fig.patch.set_facecolor('white')
# fig.show()
# # # plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_allcom.png", dpi=600)


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
# # # plt.savefig(pathCamels['Out'] + '/' + save_path + "/TS.png", dpi=500)
#
# # plot NSE map
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# data = statDictLst[0]['NSE']
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='LSTM', cRange=[0.1, 1.0], shape=None)
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# data = statDictLst[3]['NSE']
# plot.plotMap(data, ax=None, lat=gagelat, lon=gagelon, title='DA-7', cRange=[0.1, 1.0], shape=None)
# deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
# plot.plotMap(deltaNSE, ax=None, lat=gagelat, lon=gagelon, title='Delta NSE', shape=None)
# plt.savefig(pathCamels['Out'] + '/' + save_path + "/DeltaNSE.png", dpi=600)
# # plot multiple NSE maps
# plt.rcParams['font.size'] = 13
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# fig = plt.figure(figsize=(12,8), tight_layout=True)
# for ii in range(len(statDictLst)):
#     data = statDictLst[ii]['NSE']
#     ax = plt.subplot(3, 2, ii + 1)
#     if ii == 0:
#         subtitle = 'LSTM'
#     else:
#         subtitle = 'DA-'+str(nDayLst[ii-1])
#     mm, cs=plot.plotMap(data, ax=ax, lat=gagelat, lon=gagelon, title=subtitle,
#                  cRange=[0.0, 1.0], shape=None, clbar=False)
# cax = plt.axes([0.25, 0.05, 0.5, 0.02])
# plt.colorbar(cs, cax=cax,orientation="horizontal")
# plt.tight_layout()
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/NSEMap.png", dpi=600)
#
# # plot multiple NSE maps
# plt.rcParams['font.size'] = 13
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# fig, axs = plt.subplots(3,2, figsize=(12,8), constrained_layout=True)
# axs = axs.flat
# for ii in range(len(axs)):
#     data = statDictLst[ii]['NSE']
#     ax = axs[ii]
#     if ii == 0:
#         subtitle = 'LSTM'
#     else:
#         subtitle = 'DA-'+str(nDayLst[ii-1])
#     mm, cs=plot.plotMap(data, ax=ax, lat=gagelat, lon=gagelon, title=subtitle,
#                  cRange=[0.0, 1.0], shape=None, clbar=False)
# fig.colorbar(cs, ax = axs, shrink=0.7)
# # plt.tight_layout()
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/NSEMap3.png", dpi=600)
#
# # plot TS map
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# t = utils.time.tRange2Array(tRange)
# tsdata=list()
# tsdata.append(obs.squeeze())
# npred=2
# for ii in range(npred):
#     tsdata.append(predLst[ii].squeeze())
# deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
# plot.plotTsMap(dataMap=deltaNSE, dataTs=tsdata, t=t, lat=gagelat, lon=gagelon,
#                tsNameLst=['USGS', 'LSTM', 'DA-1'], figsize=[10, 8])
#
# # plot timeseries and locations
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# plt.rcParams['font.size'] = 14
# gageindex = [352, 271, 641, 374, 482, 292, 475]
# plat = gagelat[gageindex]
# plon = gagelon[gageindex]
# t = utils.time.tRange2Array(tRange)
# fig, axes = plt.subplots(4,2, figsize=(12,10), constrained_layout=True)
# axes = axes.flat
# npred = 2
# subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(k)', '(l)']
# txt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k',]
# for k in range(len(gageindex)):
#     iGrid = gageindex[k]
#     yPlot = [obs[iGrid, :]]
#     for y in predLst[0:npred]:
#         yPlot.append(y[iGrid, :])
#     NSE_LSTM = str(round(statDictLst[0]['NSE'][iGrid], 2))
#     NSE_DA1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
#     plot.plotTS(
#         t,
#         yPlot,
#         ax=axes[k],
#         cLst='kbrmg',
#         markerLst='---',
#         legLst=['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DA1], title=subtitle[k], linespec=['-',':',':'])
# plot.plotlocmap(plat, plon, ax=axes[-1], baclat=gagelat, baclon=gagelon, title=subtitle[-1], txtlabel=txt)
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/Timeseriesmap_merge.png", dpi=600)
#
# # write evaluation results
# mFile = os.path.join(pathCamels['Out'], save_path, 'evaluation.npy')
# np.save(mFile, statDictLst)
#
# obsFile = os.path.join(pathCamels['Out'], save_path, 'obs.npy')
# np.save(obsFile, obs)
#
# predFile = os.path.join(pathCamels['Out'], save_path, 'pred.npy')
# np.save(predFile, predLst)
#
# # plot the comparison of DI in different time scales
# # temporatily test
# fname_ANN = pathCamels['Out'] + '/comparison/ANN/hid1_256hid2_256batch_36500epoch_200/evaluation.npy'
# stadic_ann = np.load(fname_ANN, allow_pickle=True).item()
# fname_Auto = pathCamels['Out'] + '/comparison/Autoreg/evaluation.npy'
# stadic_auto = np.load(fname_Auto, allow_pickle=True).item()
# meanfile = '/longtermDA/testmvaverage/epochs200_batch100_rho365_hiddensize256'
# fname_damean = pathCamels['Out'] + meanfile + '/evaluation.npy'
# stadic_damean= np.load(fname_damean, allow_pickle=True).tolist()
# multifile = '/longtermDA/testmultiobs/epochs200_batch100_rho365_hiddensize256'
# fname_damulti = pathCamels['Out'] + multifile + '/evaluation.npy'
# stadic_damulti= np.load(fname_damulti, allow_pickle=True).tolist()
# keyLst=['Bias', 'NSE']
# dataBox = list()
# plotDictLst = [statDictLst[0], statDictLst[1], stadic_damean[1], stadic_damulti[1], stadic_auto, stadic_ann]
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(plotDictLst)):
#         data = plotDictLst[k][statStr]
#         data = data[~np.isnan(data)]
#         temp.append(data)
#     dataBox.append(temp)
# labelname = ['LSTM', 'DI(1)', 'DI(3)-M', 'DI(3)-A', 'ARp(1)', 'ANN(1)']
# xlabel = ['Bias ($\mathregular{ft^3}$/s)', 'NSE']
# fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 4))
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_tempcom.png", dpi=600)
#
# # test plot
# keyLst=['Bias', 'NSE']
# dataBox = list()
# plotDictLst = [statDictLst[1:], stadic_damean[1:], stadic_damulti[1:]]
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(plotDictLst)):
#         tempdict = plotDictLst[k]
#         tempvar = list()
#         for ii in range(len(tempdict)):
#             data = tempdict[ii][statStr]
#             data = data[~np.isnan(data)]
#             tempvar.append(data)
#         temp.append(tempvar)
#     dataBox.append(temp)
# p1 = [1] + np.arange(1.75, 6, 1).tolist()
# p2 = np.arange(2.0, 7.0, 1).tolist()
# p3 = np.arange(2.25, 7.0, 1).tolist()
# positions = [p1, p2, p3]
# labelname = ['DI(N)', 'DI(N)-M', 'DI(N)-A']
# ylabel = ['Bias ($\mathregular{ft^3}$/s)', 'NSE']
# xlabel = ['Days', 'Days']
# xticks = [1, 3, 10, 30, 100, 365]
# xticks = list(map(str, xticks))
# fig = plot.plotMultiBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(12, 5), position=positions,
#                            xticklabel=xticks, ylabel=ylabel)
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_comdi3.png", dpi=600)
#
# # spatial pattern comparison
# plt.rcParams['font.size'] = 13
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# fig, axs = plt.subplots(3,1, figsize=(8,8), constrained_layout=True)
# axs = axs.flat
# ComDictLst = [statDictLst[1], stadic_damean[1], stadic_damulti[1]]
# subtitles = ['(a) DI(3)-M - DI(1)', '(b) DI(3)-A - DI(1)', '(c) DI(3)-A - DI(3)-M']
# for ii in range(len(axs)):
#     if ii == 2:
#         data = ComDictLst[2]['NSE']-ComDictLst[1]['NSE']
#     else:
#         data = ComDictLst[ii+1]['NSE']-ComDictLst[0]['NSE']
#     ax = axs[ii]
#     mm, cs=plot.plotMap(data, ax=ax, lat=gagelat, lon=gagelon, title=subtitles[ii], shape=None, clbar=False,
#                         cRange=[-0.05, 0.05])
# fig.colorbar(cs, ax = axs, shrink=0.8)
# # plt.tight_layout()
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/NSEpattern_comdi3.eps", format='eps', dpi=500)
#
# # plot comparison with Autoreg and ANN
# fname_ANN = pathCamels['Out'] + '/comparison/ANN/hid1_256hid2_256batch_36500epoch_200/evaluation.npy'
# stadic_ann = np.load(fname_ANN, allow_pickle=True).item()
# stadic_ann['Bias'] = stadic_ann['Bias']*0.0283168
# fname_Auto = pathCamels['Out'] + '/comparison/Autoreg/evaluation.npy'
# stadic_auto = np.load(fname_Auto, allow_pickle=True).item()
# stadic_auto['Bias'] = stadic_auto['Bias']*0.0283168
# plt.rcParams['font.size'] = 14
# keyLst=['Bias', 'NSE', 'FLV', 'FHV']
# dataBox = list()
# comdict = [stadic_auto, stadic_ann]
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(statDictLst)):
#         data = statDictLst[k][statStr]
#         data = data[~np.isnan(data)]
#         data = data[~np.isinf(data)]
#         temp.append(data)
#     # if statStr == 'Bias' or statStr == 'NSE':
#     for jj in range(len(comdict)):
#         data = comdict[jj][statStr]
#         data = data[~np.isnan(data)]
#         data = data[~np.isinf(data)]
#         temp.append(data)
#     dataBox.append(temp)
# labelname = ['LSTM']
# for nDay in nDayLst:
#     labelname.append('DI('+str(nDay)+')')
# labelname = labelname + ['$\mathregular{AR_B(1)}$', 'ANN(1)']
# subindex = ['(a)', '(b)', '(c)', '(d)']
# ylabel = ['Bias ($\mathregular{m^3}$/s)', 'NSE', 'FLV(%)', 'FHV(%)']
# fig = plot.plotBoxF(dataBox, sharey=False, figsize=(12, 8),
#                            xticklabel=labelname, ylabel=ylabel, subtitles=subindex)
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_Figure1_final.png", dpi=600)
#
# # plot the results of DI in different scales (Flexibility)
# mvfile = '/longtermDA/testmvaverage/epochs200_batch100_rho365_hiddensize256'
# fname_damv = pathCamels['Out'] + mvfile + '/evaluation.npy'
# stadic_damv= np.load(fname_damv, allow_pickle=True).tolist()
# multifile = '/longtermDA/testmultiobs_rerun/epochs200_batch100_rho365_hiddensize256/results'
# fname_damulti = pathCamels['Out'] + multifile + '/evaluation.npy'
# stadic_damulti= np.load(fname_damulti, allow_pickle=True).tolist()
# rmfile = '/longtermDA/testaverage/epochs200_batch100_rho365_hiddensize256'
# fname_darm = pathCamels['Out'] + rmfile + '/evaluation.npy'
# stadic_darm = np.load(fname_darm, allow_pickle=True).tolist()
# keyLst=['NSE']
# dataBox = list()
# plotDictLst = [[statDictLst[0]], statDictLst[1:], stadic_darm[1:], stadic_damv[1:], stadic_damulti[1:]]
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     temp = list()
#     for k in range(len(plotDictLst)):
#         tempdict = plotDictLst[k]
#         tempvar = list()
#         for ii in range(len(tempdict)):
#             data = tempdict[ii][statStr]
#             data = data[~np.isnan(data)]
#             tempvar.append(data)
#         temp.append(tempvar)
#     dataBox.append(temp)
# p0 = [0.0]
# p1 = [1] + np.arange(1.625, 6, 1).tolist()
# p2 = np.arange(1.875, 6.0, 1).tolist()
# p3 = np.arange(2.125, 7.0, 1).tolist()
# p4 = np.arange(2.375, 7.0, 1).tolist()
# positions = [p0, p1, p2, p3, p4]
# labelname = ['LSTM', 'DI(N)', 'DI(N)-R', 'DI(N)-M', 'DI(N)-A',]
# ylabel = ['NSE']
# xlabel = ['Days']
# xticks = [1, 3, 7, 30, 100, 365]
# xticks = list(map(str, xticks))
# xticks = ['LSTM'] + xticks
# fig = plot.plotMultiBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(10, 5), position=positions,
#                            xticklabel=xticks, ylabel=ylabel, colorLst='rbkcmy',)
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/boxstat_allinone_rev.png", dpi=600)
#
# # plot multiple NSE maps
# plt.rcParams['font.size'] = 13
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# nDayLst = [1, 3, 7, 31, 53]
# fig, axs = plt.subplots(3,1, figsize=(8,8), constrained_layout=True)
# axs = axs.flat
# data = statDictLst[0]['NSE']
# plot.plotMap(data, ax=axs[0], lat=gagelat, lon=gagelon, title='(a) LSTM', cRange=[0.0, 1.0], shape=None)
# data = statDictLst[1]['NSE']
# plot.plotMap(data, ax=axs[1], lat=gagelat, lon=gagelon, title='(b) DI(1)', cRange=[0.0, 1.0], shape=None)
# deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
# plot.plotMap(deltaNSE, ax=axs[2], lat=gagelat, lon=gagelon, title='(c) Delta NSE', shape=None)
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/Figure_DeltaNSE.png", dpi=600)
#
# # plot timeseries of two typical regions
# plt.rcParams['font.size'] = 12
# gageindex = [292, 475]
# plat = gagelat[gageindex]
# plon = gagelon[gageindex]
# t = utils.time.tRange2Array(tRange)
# fig, axes = plt.subplots(2,1, figsize=(8,5), constrained_layout=True)
# axes = axes.flat
# npred = 2
# subtitle = ['(a) North Great Plain', '(b) South Texas']
# # txt = ['a', 'b', 'c', 'd', 'e']
# for k in range(len(gageindex)):
#     iGrid = gageindex[k]
#     yPlot = [obs[iGrid, :]]
#     for y in predLst[0:npred]:
#         yPlot.append(y[iGrid, :])
#     NSE_LSTM = str(round(statDictLst[0]['NSE'][iGrid], 2))
#     NSE_DA1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
#     plot.plotTS(
#         t,
#         yPlot,
#         ax=axes[k],
#         cLst='kbrmg',
#         markerLst='---',
#         legLst=['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DA1], title=subtitle[k], linespec=['-',':',':'])
# # plot.plotlocmap(plat, plon, ax=axes[-1], baclat=gagelat, baclon=gagelon, title=subtitle[-1], txtlabel=txt)
# fig.patch.set_facecolor('white')
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/typical_Timeseries.png", dpi=600)
#
# # plot new multiple NSE maps-longer time scales
# plt.rcParams['font.size'] = 13
# gageinfo = camels.gageDict
# gagelat = gageinfo['lat']
# gagelon = gageinfo['lon']
# fig, axs = plt.subplots(2,2, figsize=(12,6), constrained_layout=True)
# axs = axs.flat
# plotdic = statDictLst[2:6]
# plotnday = nDayLst[1:5]
# subtitles = ['(a)', '(b)', '(c)', '(d)']
# for ii in range(len(axs)):
#     data = plotdic[ii]['NSE']
#     ax = axs[ii]
#     subtitle = subtitles[ii] + ' DI('+str(plotnday[ii])+')'
#     mm, cs=plot.plotMap(data, ax=ax, lat=gagelat, lon=gagelon, title=subtitle,
#                  cRange=[0.0, 1.0], shape=None, clbar=False)
# fig.colorbar(cs, ax = axs, shrink=0.7)
# # plt.tight_layout()
# fig.show()
# # plt.savefig(pathCamels['Out'] + '/' + save_path + "/NSEMap_longer.png", dpi=600)
