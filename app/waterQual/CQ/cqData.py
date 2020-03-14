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
from scipy.optimize import curve_fit
from scipy.stats import linregress

import importlib

if False:
    # load data - processed in slopeCal.py
    dirCQ = os.path.join(kPath.dirWQ, 'C-Q')
    fileName = os.path.join(dirCQ, 'CQall')
    dictData = pickle.load(open(fileName, 'rb'))
    dfSa = pd.read_csv(os.path.join(dirCQ, 'slope_a'), dtype={
        'siteNo': str}).set_index('siteNo')
    dfSb = pd.read_csv(os.path.join(dirCQ, 'slope_b'), dtype={
        'siteNo': str}).set_index('siteNo')
    dfCeq = pd.read_csv(os.path.join(dirCQ, 'kate_ceq'), dtype={
        'siteNo': str}).set_index('siteNo')
    dfDw = pd.read_csv(os.path.join(dirCQ, 'kate_dw'), dtype={
        'siteNo': str}).set_index('siteNo')
    dfN = pd.read_csv(os.path.join(dirCQ, 'nSample'), dtype={
        'siteNo': str}).set_index('siteNo')
    siteNoLst = dfN.index.tolist()
    codeLst = dfN.columns.tolist()
    dfPLst = [dfSa, dfSb, dfCeq, dfDw]
    strPLst = ['slope-a', 'slope-b', 'ceq', 'dw']
    pdfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
    unitConv = 0.3048**3*365*24*60*60/1000**2
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)

dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoSel= [siteNo for siteNo in dfHBN.index.tolist() if siteNo in siteNoLst]

# dfNsel = dfN[codeSel]
# siteNoSel = dfNsel[(dfNsel > 100).all(axis=1)].index.tolist()

# plot click map
codeSel = ['00955', '00940', '00915']
nCode = len(codeSel)
importlib.reload(axplot)
codePdf = waterQuality.codePdf
lat = dfCrd['LAT_GAGE'][siteNoSel].values
lon = dfCrd['LNG_GAGE'][siteNoSel].values
figM, axM = plt.subplots(nCode, 2, figsize=(8, 6))
for j, code in enumerate(codeSel):
    for i, (dfP, strP) in enumerate(zip([dfSb, dfDw], ['slope', 'Dw'])):
        dataMap = dfP[code][siteNoSel].values
        strTitle = '{} of {} '.format(strP, codePdf['srsName'][code])
        vr = np.max([np.abs(np.percentile(dataMap[~np.isnan(dataMap)], 10)),
                     np.abs(np.percentile(dataMap[~np.isnan(dataMap)], 90))])
        axplot.mapPoint(axM[j, i], lat, lon, dataMap, title=strTitle,
                        vRange=[-vr, vr], s=6)

figP, axP = plt.subplots(nCode, 1, figsize=(8, 6))


def onclick(event):
    xClick = event.xdata
    yClick = event.ydata
    iP = np.argmin(np.sqrt((xClick - lon)**2 + (yClick - lat)**2))
    for temp in axM:
        for ax in temp:
            [p.remove() for p in reversed(ax.patches)]
            circle = plt.Circle([lon[iP], lat[iP]], 1,
                                color='black', fill=False)
            ax.add_patch(circle)

    siteNo = siteNoSel[iP]
    area = pdfArea.loc[siteNo].values[0]
    q = dictData[siteNo]['00060_00003'].values/area*unitConv
    for k, code in enumerate(codeSel):
        axP[k].clear()
        c = dictData[siteNo][code].values
        x = 10**np.linspace(np.log10(np.min(q[q > 0])),
                            np.log10(np.max(q[~np.isnan(q)])), 20)
        sa = dfSa[code][siteNo]
        sb = dfSb[code][siteNo]
        ceq = dfCeq[code][siteNo]
        dw = dfDw[code][siteNo]
        ys = sa*x**sb
        yk = ceq/(1+x/dw)
        axP[k].plot(np.log10(q), c, '*k',  label='obs')
        axP[k].plot(np.log10(x), ys, '-b',
                    label='{:.2f} q ^ {:.2f}'.format(sa, sb))
        axP[k].plot(np.log10(x), yk, '-r',
                    label='{:.2f} 1/(q/{:.2f}+1)'.format(ceq, dw))
        axP[k].legend()
    axP[0].set_title(siteNo)
    figM.canvas.draw()
    figP.canvas.draw()


figM.canvas.mpl_connect('button_press_event', onclick)
figP.show()
figM.show()
