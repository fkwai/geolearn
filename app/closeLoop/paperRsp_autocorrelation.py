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
    tRange = [20160401, 20180401]
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA2015')
    df, yf, obs = master.test(out, tRange=tRange, subset=subset, batchSize=100)
    out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM2015')
    df, yp, obs = master.test(out, tRange=tRange, subset=subset)
    yf = yf.squeeze()
    yp = yp.squeeze()
    obs = obs.squeeze()
    ym = df.getDataTs(varLst='SOILM_0-10_NOAH', doNorm=False)
    ym = ym.squeeze()/100


def autocorr(x, lag):
    a = x[lag:]
    b = x[:-lag]
    ind = np.where(~np.isnan(a) & ~np.isnan(b))
    corr = np.corrcoef(a[ind], b[ind])[0, 1]
    return(corr)


ngrid, nt = yf.shape
mask = np.ones([ngrid, nt])
mask[np.isnan(obs)] = np.nan
dataLst = [obs, yf*mask, yp*mask, ym*mask]
lagLst = [1, 2, 3]
acLst = list()
for k in range(len(dataLst)):
    ac = np.zeros([ngrid, 3])
    for lag in lagLst:
        for igrid in range(ngrid):
            corr = autocorr(dataLst[k][igrid, :], lag)
            corr = np.nan if corr > 0.99 else corr
            corr = np.nan if corr == 0 else corr
            ac[igrid, lag-1] = corr
    acLst.append(ac)

fig, axes = plt.subplots(3, 3)
modelLst = ['DI-LSTM', 'LSTM', 'Noah']
for j in range(3):
    for i in range(3):
        titleStr = modelLst[j]+' {}d-lag'.format(i+1)
        ax = axes[j, i]
        plot.plotVS(acLst[0][:, i], acLst[j+1][:, i],
                    ax=ax, title=titleStr)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'autocorrRho'))




matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})
matplotlib.rcParams.update({'legend.fontsize': 11})
fig, axes = plt.subplots(3, 3,figsize=(8,8))
modelLst = ['DI-LSTM', 'LSTM', 'Noah']
for j in range(3):
    for i in range(3):
        titleStr = modelLst[j]+' {}d-lag'.format(i+1)
        ax = axes[j, i]
        plot.plotVS(acLst[0][:, i], acLst[j+1][:, i],
                    ax=ax, titleCorr=False)
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 0.5, 1])
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1])
        if j!=2:
            ax.set_xticklabels([])
        if i!=0:
            ax.set_yticklabels([])
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveDir, 'autocorr.eps'))
fig.savefig(os.path.join(saveDir, 'autocorr'))
