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
outLst = [
    'CONUSv2f1_DA2015', 'CONUSv2f1_DA_2015RK', 'CONUSv2f1_DA_Prcp_2015',
    'CONUSv2f1_DA_Prcp_2015RK'
]
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
        statErr = stat.statError(utils.fillNan(yf, maskF),
                                 utils.fillNan(obs, maskF))
        temp.append(statErr)
    statLst.append(temp)

# load result from RK
dirRK = r'D:\\data\\Koster17\\'
fileNameLst = ['rmse_lead_{}.dat'.format(x) for x in [1, 2, 3]]
tempLst = list()
for k in range(3):
    # lon lat are identical. Tested
    temp = np.loadtxt(os.path.join(dirRK, fileNameLst[k]))
    tempLst.append(temp[:, 2])
RKlon = temp[:, 0]
RKlat = temp[:, 1]
lat, lon = df.getGeo()
errLst = list()
RKrmseMat = np.zeros([len(lat), 3]) * np.nan
for k in range(len(lat)):
    y = round(lat[k], 4)
    x = round(lon[k], 4) if lon[k] > -100 else round(lon[k], 3)
    ind = np.where(
        np.isclose(RKlat, lat[k], rtol=0, atol=0.01)
        & np.isclose(RKlon, lon[k], rtol=0, atol=0.01))[0]
    if len(ind) > 1:
        errLst.append(ind)
    if len(ind) > 0:
        for kk in range(3):
            if tempLst[kk][ind]==0:
                RKrmseMat[k, kk]=np.nan
            else:
                RKrmseMat[k, kk] = tempLst[kk][ind]

# plot box - forecast
import importlib
importlib.reload(plot)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 12})
caseLst = [
    'DI-LSTM All Forcing, \n2015/04/01-2016/03/31',
    'DI-LSTM All Forcing, \n2015/05/01-2015/09/31',
    'DI-LSTM Prcp only, \n2015/04/01-2016/03/31', 'DI-LSTM Prcp only, \n2015/05/01-2015/09/31',
    'K17 Prcp only, \n2015/05/01-2015/09/31'
]
xLst = ['{}d Forecast'.format(x) for x in fLst]
key = 'RMSE'
dataBox = list()
for i in range(len(statLst)):
    ss = statLst[i]
    RKrmse = RKrmseMat[:, i]
    ind = np.where(~np.isnan(RKrmse))[0]
    temp = list()
    for s in ss:
        temp.append(s[key][ind])
    temp.append(RKrmse[ind])
    dataBox.append(temp)
fig = plot.plotBoxFig(
    dataBox,
    xLst,
    None,
    sharey=True,
    figsize=[6, 4],
    title='RMSE of forecast models using different training set')
fig.show()
fig.savefig(os.path.join(saveDir, 'compareRK.eps'))
fig.savefig(os.path.join(saveDir, 'compareRK.png'))

importlib.reload(plot)
fig = plot.plotBoxFig(dataBox,
                      xLst,
                      caseLst,
                      sharey=True,
                      figsize=[8, 4],
                      legOnly=True)
fig.show()
fig.savefig(os.path.join(saveDir, 'compareRK_leg.eps'))
fig.savefig(os.path.join(saveDir, 'compareRK_leg.png'))

for i in range(len(dataBox)):
    print('{}d forecast, mean RMSE = {:.4f}; {:.4f}; {:.4f}; {:.4f}; {:.4f}'.format(
        i+1, np.nanmean(dataBox[i][0]), np.nanmean(dataBox[i][1]),
        np.nanmean(dataBox[i][2]), np.nanmean(dataBox[i][3]),
        np.nanmean(dataBox[i][4])))
    vLst=[np.nanmean(dataBox[i][k]) for k in range(5)]
    print('{}d forecast, mean RMSE = {:.4f}; {:.4f}; {:.4f}; improve {:.2f}%'.format(
        i+1, vLst[0],vLst[3],vLst[4],(vLst[4]-vLst[3])/vLst[4]*100))
    # print('{}d forecast, median RMSE = {:.4f}; {:.4f}; {:.4f}; {:.4f}; {:.4f}'.format(
    #     i+1, np.nanmedian(dataBox[i][0]), np.nanmedian(dataBox[i][1]),
    #     np.nanmedian(dataBox[i][2]), np.nanmedian(dataBox[i][3]),
    #     np.nanmedian(dataBox[i][4])))

# map forecast
importlib.reload(plot)
[lat, lon] = df.getGeo()
fig, axes = plt.subplots(1,len(fLst), figsize=[8, 3])    
key = 'RMSE'
cRange=[-0.03,0.03]
for j in range(len(fLst)):
    data = RKrmseMat[:, j]-statLst[j][3][key]    
    titleStr=' {}d Forecast'.format(fLst[j])
    grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
    plot.plotMap(grid,
                    ax=axes[j],
                    lat=uy,
                    lon=ux,
                    title=titleStr,
                    cRange=cRange)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'map_forecast_RK.eps'))
fig.savefig(os.path.join(saveDir, 'map_forecast_RK.png'))
