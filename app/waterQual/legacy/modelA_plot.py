import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from mpl_toolkits.basemap import Basemap

from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit
from hydroDL.post import plot
from hydroDL.data import gageII
import matplotlib.pyplot as plt
from datetime import datetime as dt
from random import randint

caseName = 'refBasins'
# caseName = 'temp'
nEpoch = 500
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
dictData, info, x, y, c = waterQuality.loadData(caseName)

targetFile = os.path.join(modelFolder, 'target.csv')
dfT = pd.read_csv(targetFile, dtype={'siteNo': str})
outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
dfP = pd.read_csv(outFile, dtype={'siteNo': str})

siteNoLst = dictData['siteNoLst']
varC = dictData['varC']
statRes = np.load(os.path.join(
    modelFolder, 'statResult_Ep{}.npz'.format(nEpoch)))
nameC = ['Calcium [mg/l]',
         'Magnesium [mg/l]',
         'Sodium [mg/l]',
         'Potassium [mg/l]',
         'Silica [mg/l]',
         'Chloride [mg/l]',
         'Sulfate [mg/l]',
         'Alkalinity [mg/l]',
         'Alkalinity [mg/l]',
         'Alkalinity [mg/l]',
         'Alkalinity [mg/l]',
         'Oxygen [percent]',
         'Oxygen [mg/l]',
         'Nitrate [mg/l]',
         'Organic carbon [mg/l]',
         'Phosphate-phosphorus [mg/l]',
         'Temperature, water [C]',
         'Total suspended solids [mg/l]',
         'Specific conductance [uS/cm]',
         'pH unfiltered',
         'pH filtered']

# plot map
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values

icLst = range(len(varC))
for j, strTrain in enumerate(['Train', 'Test']):
    dataStrLst = ['matRho'+str(j+1), 'matRmse'+str(j+1)]
    dataStrN = 'matN'+str(j+1)
    titleD = ['correlation', 'RMSE']
    figFolder = os.path.join(modelFolder, 'fig_Ep{}'.format(nEpoch))
    if not os.path.exists(figFolder):
        os.mkdir(figFolder)
    for i, iC in enumerate(icLst):
        fig, axes = plt.subplots(2, 1, figsize=[6, 6.5])
        for k, dataStr in enumerate(dataStrLst):
            data = statRes[dataStr][:, iC]
            dataN = statRes[dataStrN][:, iC]
            if np.isnan(data).all():
                continue
            if k == 0:
                vmin = 0
                vmax = 1
            else:
                vmin = np.percentile(data[~np.isnan(data)], 10)
                vmax = np.percentile(data[~np.isnan(data)], 90)
            mm = Basemap(llcrnrlat=25, urcrnrlat=50,
                         llcrnrlon=-125, urcrnrlon=-65,
                         projection='cyl', resolution='c', ax=axes[k])
            mm.drawcoastlines()
            mm.drawstates(linestyle='dashed')
            ind1 = np.where(dataN >= 10)
            ind2 = np.where(dataN < 10)
            cs = mm.scatter(lon[ind1], lat[ind1], c=data[ind1], cmap=plt.cm.jet,
                            s=80, marker='.', vmin=vmin, vmax=vmax)
            cs = mm.scatter(lon[ind2], lat[ind2], c=data[ind2], cmap=plt.cm.jet,
                            s=30, marker='*', vmin=vmin, vmax=vmax)
            mm.colorbar(cs, location='bottom', pad='5%')
            axes[k].set_title('{}ing {} of {} ({})'.format(
                strTrain, titleD[k], nameC[i], varC[i]))

        # plt.tight_layout
        # fig.show()
        saveName = 'map_{}_{}'.format(varC[i], strTrain)
        plt.savefig(os.path.join(figFolder, saveName))
