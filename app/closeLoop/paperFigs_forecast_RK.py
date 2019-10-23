from hydroDL import pathSMAP, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

doLst = list()
# doLst.append('train')
doLst.append('test')
doLst.append('post')
saveDir = os.path.join(pathSMAP['dirResult'], 'DA', 'paper')

# test
if 'test' in doLst:
    torch.cuda.set_device(2)
    subset = 'CONUSv2f1'
    tRange = [20160501, 20161001]
    
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA_Prcp_2015RK')
    df, yf, obs = master.test(out, tRange=tRange, subset=subset)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA_Prcp_2015')
    df, yp, obs = master.test(out, tRange=tRange, subset=subset)
    yf = yf.squeeze()
    yp = yp.squeeze()
    obs = obs.squeeze()

    # tRangeTest = [20160501, 20161001]
    # t1 = utils.time.tRange2Array(tRange)
    # t2 = utils.time.tRange2Array(tRangeTest)
    # ind1, ind2 = utils.time.intersect(t1, t2)
    # yf = yf[:, ind1]
    # yp = yp[:, ind1]
    # obs = obs[:, ind1]


# add quality flag
# qualRec = df.getDataTs(varLst='qual_recommend_AM',doNorm=False,rmNan=False)
# qualSuc = df.getDataTs(varLst='qual_successful_AM', doNorm=False, rmNan=False)
# qualAtp = df.getDataTs(varLst='qual_attempt_AM', doNorm=False, rmNan=False)
# qualLst = [qualRec, qualSuc, qualAtp]
# maskObs = 1 * ~np.isnan(obs.squeeze())
# [lat, lon] = df.getGeo()
# fig, axes = plt.subplots(len(qualLst), 1, figsize=[8, 7])
# for k in range(len(qualLst)):    
#     temp = (qualLst[k].squeeze()==0)*1    
#     data = np.sum(temp, axis=1)/np.sum(maskObs,axis=1)    
#     grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
#     plot.plotMap(
#         grid, ax=axes[k], lat=uy, lon=ux)
# plt.tight_layout()
# fig.show()

# grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
# plot.plotMap(grid, ax=axes[k], lat=uy, lon=ux)
# plt.tight_layout()
# fig.show()

# flag = df.getDataConst(varLst='flag_extraOrd',  doNorm=False, rmNan=False)

# figure out how many days observation lead
# maskObs = 1 * ~np.isnan(obs.squeeze()) * (qualSuc.squeeze() == 0)
maskObs = 1 * ~np.isnan(obs.squeeze()) * (qualSuc.squeeze() == 0) * np.repeat(flag==0,153,axis=1)
maskDay = np.zeros(maskObs.shape).astype(int)
ngrid, nt = maskObs.shape
for j in range(ngrid):
    temp = 0
    for i in range(nt):
        maskDay[j, i] = temp
        if maskObs[j, i] == 1:
            temp = 1
        else:
            if temp != 0:
                temp = temp + 1
ind = np.random.randint(0, ngrid)
print(np.array([maskObs[ind, :], maskDay[ind, :]]))
maskObsDay = maskObs * maskDay
unique, counts = np.unique(maskDay, return_counts=True)
print(np.asarray((unique, counts)).T)
print(counts / ngrid / nt)


fLst = [1, 2, 3]
statLstF = list()
statLstP = list()
maskF = (maskDay >= 1) & (maskDay <= 3)
statP = stat.statError(utils.fillNan(yp, maskF), utils.fillNan(obs, maskF))
statF = stat.statError(utils.fillNan(yf, maskF), utils.fillNan(obs, maskF))
for nf in fLst:
    xp = np.full([ngrid, nt], np.nan)
    xf = np.full([ngrid, nt], np.nan)
    y = np.full([ngrid, nt], np.nan)
    xf[maskObsDay == nf] = yf[maskObsDay == nf]
    xp[maskObsDay == nf] = yp[maskObsDay == nf]
    y[maskObsDay == nf] = obs[maskObsDay == nf]
    statLstF.append(stat.statError(xf, y))
    statLstP.append(stat.statError(xp, y))
    
# fig,axes=plt.subplots(2, 1, figsize=[8, 7])
# data = np.nansum((maskDay == 1)*1, axis=1)
# # data[data==0]=-999
# grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
# plot.plotMap(grid,ax=axes[0], lat=uy, lon=ux)

# data = statLstF[0]['RMSE']
# grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
# plot.plotMap(grid, ax=axes[1], lat=uy, lon=ux)
# fig.show()

# plot box - forecast
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 11})
keyLst = stat.keyLst
keyLegLst = ['Bias', 'RMSE', 'ubRMSE', 'R']
dataBox = list()
caseLst = ['Project'] + [str(nd) + 'd Forecast' for nd in fLst]
for k in range(len(keyLst)):
    key = keyLst[k]
    temp = list()
    data = statP[key]
    temp.append(data)
    for i in range(len(fLst)):
        data = statLstF[i][key]
        temp.append(data)
    dataBox.append(temp)
fig = plot.plotBoxFig(dataBox, keyLegLst, sharey=False, figsize=[8, 4])
plt.suptitle('Error metrics of projection and forecast model')
plt.tight_layout()
plt.subplots_adjust(top=0.85, right=0.95)
fig.show()
fig.savefig(os.path.join(saveDir, 'box_forecast_RK.eps'))
fig.savefig(os.path.join(saveDir, 'box_forecast_RK.png'))

[np.nanmean(dataBox[1][1]), np.nanmean(
    dataBox[1][2]), np.nanmean(dataBox[1][3])]

# fig = plot.plotBoxFig(
#     dataBox, keyLst, caseLst, sharey=False, figsize=[8, 3], legOnly=True)
# # plt.suptitle('Error matrices of project and forecast model')
# plt.tight_layout()
# fig.show()
# fig.savefig(os.path.join(saveDir, 'box_forecast_leg.eps'))
# fig.savefig(os.path.join(saveDir, 'box_forecast_leg.png'))

# map forecast
keyLst = ['RMSE', 'Corr']
cRangeLst = [[0, 0.1], [0., 1]]
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(len(fLst), len(keyLst), figsize=[8, 7])
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    for j in range(len(fLst)):
        data = statLstF[j][key]
        if key == 'Corr':
            titleStr = 'R of {}d Forecast'.format(fLst[j])
        else:
            titleStr = key + ' of {}d Forecast'.format(fLst[j])
        grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
        plot.plotMap(
            grid, ax=axes[j][i], lat=uy, lon=ux, title=titleStr, cRange=cRange)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_forecast_RK.eps'))
fig.savefig(os.path.join(saveDir, 'map_forecast_RK.png'))

[np.nanmean(dataBox[1][1]), np.nanmean(
    dataBox[1][2]), np.nanmean(dataBox[1][3])]
