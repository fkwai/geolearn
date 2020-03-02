from hydroDL.data import usgs, gageII, transform
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.post import axplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import pickle
from scipy.stats import linregress
import importlib

if True:
    # load data - processed in dataProcess.py
    dirCQ = os.path.join(kPath.dirWQ, 'C-Q')
    fileName = os.path.join(dirCQ, 'CQall')
    dictData = pickle.load(open(fileName, 'rb'))
    dfS = pd.read_csv(os.path.join(dirCQ, 'slopeLog'), dtype={
        'siteNo': str}).set_index('siteNo')
    dfN = pd.read_csv(os.path.join(dirCQ, 'nSample'), dtype={
        'siteNo': str}).set_index('siteNo')
    siteNoLst = dfS.index.tolist()
    codeLst = dfS.columns.tolist()
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)

# plot click map
codeSel = ['00955', '00940', '00915']
nCode = len(codeSel)
importlib.reload(axplot)
codePdf = waterQuality.codePdf
dfNsel = dfN[codeSel]
siteNoSel = dfNsel[(dfNsel > 100).all(axis=1)].index.tolist()
lat = dfCrd['LAT_GAGE'][siteNoSel].values
lon = dfCrd['LNG_GAGE'][siteNoSel].values
figM, axM = plt.subplots(nCode,1, figsize=(8, 6))
for i, code in enumerate(codeSel):
    print(i)
    slopeAry = dfS[code].values
    dataMap = dfS[code][siteNoSel].values
    strTitle = 'slope of {} '.format(codePdf['srsName'][code])
    vr = np.max([np.abs(np.percentile(dataMap, 5)),
                 np.abs(np.percentile(dataMap, 95))])
    axplot.mapPoint(axM[i], lat, lon, dataMap, title=strTitle,
                    vRange=[-vr, vr], s=15)
figP, axP = plt.subplots(3, nCode, figsize=(8, 6))


def onclick(event):
    xClick = event.xdata
    yClick = event.ydata
    iP = np.argmin(np.sqrt((xClick - lon)**2 + (yClick - lat)**2))
    for ax in axM:
        [p.remove() for p in reversed(ax.patches)]
        circle = plt.Circle([lon[iP], lat[iP]], 1, color='black', fill=False)
        ax.add_patch(circle)

    siteNo = siteNoSel[iP]
    q = dictData[siteNo]['00060_00003'].values
    for k, code in enumerate(codeSel):
        c = dictData[siteNo][code].values
        axP[0, k, ].clear()
        axP[1, k, ].clear()
        axP[2, k].clear()
        axP[0, k].plot(q, c, '*')
        axP[1, k].plot(np.log10(q), c, '*')
        axP[2, k].plot(np.log10(q), np.log10(c), '*')

    figM.canvas.draw()
    figP.canvas.draw()


figM.canvas.mpl_connect('button_press_event', onclick)
figP.show()
figM.show()
