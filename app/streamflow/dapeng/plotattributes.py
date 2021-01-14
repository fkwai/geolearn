from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
import numpy as np
import os
import scipy.stats as stats
import pandas as pd

gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
plotattri = ['slope_mean', 'soil_depth_statsgo', 'aridity', 'frac_snow', 'p_seasonality','baseflow_index']
optData = default.update(default.optDataCamels, tRange=[19900101, 20000101])
df = camels.DataframeCamels(
            subset=optData['subset'], tRange=optData['tRange'])
forcing = df.getDataTs(
            varLst=optData['varT'],
            doNorm=False,
            rmNan=False)
obs = df.getDataObs(
            doNorm=False, rmNan=False)
attributes = df.getDataConst(
            varLst=plotattri,
            doNorm=False,
            rmNan=False)

def auto_corr(x, lag):
   x1 = x[0:-lag]
   x2 = x[lag:]
   ind = np.where(np.logical_and(~np.isnan(x1), ~np.isnan(x2)))[0]
   xx = x1[ind]
   yy = x2[ind]
   Corr = stats.pearsonr(xx, yy)[0]
   return Corr


# calculate the AR(1) coefficent
ngage = obs.shape[0]
autor = np.full(ngage, np.nan)
for ii in range(ngage):
    tstemp = obs[ii, :,0]
    autor[ii] = auto_corr(tstemp, 1)

attributes = np.column_stack((attributes, autor))
plotattri.append('ACF(1)')

# read gamma file
dirDB = pathCamels['DB']

datadir = os.path.join(dirDB,'camels_attributes_v2.0',
                              'camels_attributes_v2.0')
dataFile = datadir + '/Attris_Kuai.csv'
dataTemp = pd.read_csv(dataFile,header=0)
gammaall = dataTemp.Amp1.values
idFile = datadir + '/ID_Kuai.txt'
idall = np.loadtxt(idFile)
[C, ind1, ind2] = np.intersect1d(gageinfo['id'], idall, return_indices=True)
gamma = gammaall[ind2]
gammalat = gageinfo['lat'][ind1]
gammalon = gageinfo['lon'][ind1]

# plot the attributes maps
titles = ['(a) Slope', '(b) SoilDep ', '(c) Aridity', '(d) SnowFrac', r'(e) $\mathregular{\xi}$', '(f) BfInd',
          '(g) ACF(1)', r'(h) $\mathregular{\gamma}$']
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'Times New Roman'
gageinfo = camels.gageDict
gagelat = gageinfo['lat']
gagelon = gageinfo['lon']
fig, axs = plt.subplots(4,2, figsize=(14,14), tight_layout=True)
axs = axs.flat
for ii in range(attributes.shape[1]):
    data = attributes[:,ii]
    ax = axs[ii]
    subtitle = titles[ii]
    mm, cs=plot.plotMap(data, ax=ax, lat=gagelat, lon=gagelon, title=subtitle, shape=None, clbar=True)
mm, cs = plot.plotMap(gamma, ax=axs[-1], lat=gammalat, lon=gammalon, title=titles[-1], shape=None, clbar=True)
fig.show()
# plt.savefig(pathCamels['Out'] + "/basinattr_test_new.png", dpi=600)

# plot the correlation relationship
dafile = '/parameter_optim/change_basinnorm/epochs200_batch100_rho365_hiddensize256/'
fname_damean = pathCamels['Out'] + dafile + '/evaluation.npy'
stadic_da= np.load(fname_damean, allow_pickle=True).tolist()
nsemat = np.full((ngage, 3), np.nan)
nsemat[:,0] = stadic_da[0]['NSE']
nsemat[:,1] = stadic_da[1]['NSE']
nsemat[:,2] = stadic_da[1]['NSE']- stadic_da[0]['NSE']
fig, axs = plt.subplots(2,4, figsize=(13,6), constrained_layout=True)
axs = axs.flat
fmts = ['or', 'sb']
labels = ['LSTM', 'DI(1)']
for ii in range(attributes.shape[1]):
    xdata = attributes[:,ii]
    ax = axs[ii]
    # subtitle = titles[ii]
    for jj in range(len(fmts)):
        lp=ax.plot(xdata, nsemat[:,jj], fmts[jj], markerfacecolor='none', label=labels[jj])
    ax.set(xlabel=titles[ii], ylabel='NSE', ylim=[-0.5,1])
    if titles[ii] == '(c) Aridity':
        ax.set(xlim=[0, 4.0])
    if ii == 0:
        ax.legend(loc='best',handletextpad=0.1, edgecolor='black')
ax = axs[-1]
for jj in range(len(fmts)):
    lp=ax.plot(gamma, nsemat[:,jj][ind1], fmts[jj], markerfacecolor='none')
ax.set(xlabel=titles[-1], ylabel='NSE', ylim=[-0.5,1])
fig.show()
# plt.savefig(pathCamels['Out'] + "/attricorr_map_new.png", dpi=600)

# plot scatter
meanq = df.getDataConst(
            varLst='q_mean',
            doNorm=False,
            rmNan=False)
meanq = meanq*365
fig, axs = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
axs = axs.flat
# xi
ax = axs[0]
ax.scatter(meanq, attributes[:, 4], c=nsemat[:,0], cmap=plt.cm.jet, vmin=0.0, vmax=1.0)
ax.set_xlabel('Mean Annual Runoff (mm)')
ax.set_ylabel(r'$\mathregular{\xi}$')
ax.set_title('(a)', loc='left')
ax.text(x=1100, y=0.8, s='Projection LSTM', fontsize=14)
# gamma
ax = axs[1]
ss = ax.scatter(meanq[ind1, 0], gamma, c=nsemat[:,0][ind1], cmap=plt.cm.jet, vmin=0.0, vmax=1.0)
ax.set_xlabel('Mean Annual Runoff (mm)')
ax.set_ylabel(r'$\mathregular{\gamma}$')
ax.set_title('(b)', loc='left')
ax.text(x=1100, y=1.65, s='Projection LSTM', fontsize=14)
# xi DI(1)
ax = axs[2]
ax.scatter(meanq, attributes[:, 4], c=nsemat[:,1], cmap=plt.cm.jet, vmin=0.0, vmax=1.0)
ax.set_xlabel('Mean Annual Runoff (mm)')
ax.set_ylabel(r'$\mathregular{\xi}$')
ax.set_title('(c)', loc='left')
ax.text(x=1500, y=0.8, s='DI(1)', fontsize=14)
# gamma
ax = axs[3]
ss = ax.scatter(meanq[ind1, 0], gamma, c=nsemat[:,1][ind1], cmap=plt.cm.jet, vmin=0.0, vmax=1.0)
ax.set_xlabel('Mean Annual Runoff (mm)')
ax.set_ylabel(r'$\mathregular{\gamma}$')
ax.set_title('(d)', loc='left')
ax.text(x=1500, y=1.65, s='DI(1)', fontsize=14)
fig.colorbar(ss, ax = axs, shrink=0.8)
# plt.savefig(pathCamels['Out'] + "/scatterplots_xigamma.png", dpi=600)

areas = df.getDataConst(
            varLst='area_gages2',
            doNorm=False,
            rmNan=False)
