
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
# torch.cuda.set_device(2)
subset = 'CONUSv2f1'
tRange = [20160501, 20161001]
outLst = ['CONUSv2f1_DA2015', 'CONUSv2f1_DA_2015RK',
          'CONUSv2f1_DA_Prcp_2015', 'CONUSv2f1_DA_Prcp_2015RK']
yfLst = list()
for outName in outLst:
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', outName)
    df, yf, obs = master.test(out, tRange=tRange, subset=subset)
    yfLst.append(yf.squeeze())
obs = obs.squeeze()

# figure out how many days observation lead
# maskObs = 1 * ~np.isnan(obs.squeeze()) * (qualSuc.squeeze() == 0)
maskObs = 1 * ~np.isnan(obs.squeeze())
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
maskObsDay = maskObs * maskDay

fLst = [1, 2, 3]
statLst = list()
for nf in fLst:
    maskF = maskDay == nf
    temp = list()
    for yf in yfLst:
        statErr = stat.statError(utils.fillNan(
            yf, maskF), utils.fillNan(obs, maskF))
        temp.append(statErr)
    statLst.append(temp)


# plot box - forecast
import importlib
importlib.reload(plot)
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 11})
caseLst = ['All Forcing, \n2015/04/01-2016/03/31', 'All Forcing, \n2015/05/01-2015/09/31',
           'Prcp only, \n2015/04/01-2016/03/31', 'Prcp only, \n2015/05/01-2015/09/31']
xLst = ['{}d Forecast'.format(x) for x in fLst]
key = 'RMSE'
dataBox = list()
for ss in statLst:
    temp = list()
    for s in ss:
        temp.append(s[key])
    dataBox.append(temp)
fig = plot.plotBoxFig(dataBox, xLst, None, sharey=True, figsize=[6, 4],
                      title='RMSE of forecast models using different training set')
fig.show()
fig.savefig(os.path.join(saveDir, 'compareRK.eps'))
fig.savefig(os.path.join(saveDir, 'compareRK.png'))

fig = plot.plotBoxFig(dataBox, xLst, caseLst, sharey=True, figsize=[8, 4],legOnly=True)
fig.show()
fig.savefig(os.path.join(saveDir, 'compareRK_leg.eps'))
fig.savefig(os.path.join(saveDir, 'compareRK_leg.png'))

# map forecast
importlib.reload(plot)
colors = [
    (0, 0, 0.5),
    (0,0.5,1),
    (0, 0.75, 1),
    (0, 1, 1),
    (0.9, 1, 0),
    (1, 0.85, 0),
    (1, 0.75, 0),
    (1, 0.5, 0),
    (1, 0.2, 0),
    (1, 0, 0)]
cm = matplotlib.colors.LinearSegmentedColormap.from_list('temp', colors,N=10)
keyLst = ['RMSE', 'Corr']
cRangeLst = [[0, 0.1], [0., 1]]
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(len(fLst), len(keyLst), figsize=[8, 7])
for i in range(len(keyLst)):
    key = keyLst[i]
    cRange = cRangeLst[i]
    for j in range(len(fLst)):
        data = statLst[j][3][key]
        if key == 'Corr':
            titleStr = 'R of {}d Forecast'.format(fLst[j])
        else:
            titleStr = key + ' of {}d Forecast'.format(fLst[j])
        grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
        plot.plotMap(
            grid, ax=axes[j][i], lat=uy, lon=ux, title=titleStr, 
            cRange=cRange, cmap=cm,bounding=[25,50,-130,-65])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_forecast_RK.eps'))
fig.savefig(os.path.join(saveDir, 'map_forecast_RK.png'))



[np.nanmean(dataBox[0][0]),
 np.nanmean(dataBox[1][0]),
 np.nanmean(dataBox[2][0])]



[np.nanmean(dataBox[0][3]),
 np.nanmean(dataBox[1][3]),
 np.nanmean(dataBox[2][3])]


[np.nanmedian(dataBox[0][0]),
 np.nanmedian(dataBox[1][0]),
 np.nanmedian(dataBox[2][0])]

[np.nanmedian(dataBox[0][1]),
 np.nanmedian(dataBox[1][1]),
 np.nanmedian(dataBox[2][1])]

[np.nanmedian(dataBox[0][2]),
 np.nanmedian(dataBox[1][2]),
 np.nanmedian(dataBox[2][2])]

[np.nanmedian(dataBox[0][3]),
 np.nanmedian(dataBox[1][3]),
 np.nanmedian(dataBox[2][3])]
