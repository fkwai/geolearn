# initial
from hydroDL import pathSMAP, master, utils
import os
from hydroDL.data import dbCsv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from hydroDL.post import axplot, figplot, stat
import scipy
import pandas as pd

subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst1 = ['Local', 'Global']
saveFolder = r'C:\Users\geofk\work\paper\SMAP-regional'

# load data and calculate stat
statLst = list()
tRange = [20160401, 20180401]
for k in range(len(subsetLst)):
    testName = subsetLst[k]
    tempLst = list()
    for case1 in caseLst1:
        if case1 == 'Local':
            out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                               subsetLst[k] + '_Forcing')
        elif case1 == 'Global':
            out = os.path.join(pathSMAP['Out_L3_NA'], 'CONUSv2f1_Forcing')
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        temp = stat.statError(yp[:, :, 0], yt[:, :, 0])
        tempLst.append(temp)
    statLst.append(tempLst)


# plot box
keyLst = ['RMSE', 'Corr']
rangeLst = [[0, 0.08], [0.3, 1]]
# ecoLst = ['{0:0>2}'.format(x) for x in range(1, 18)]
ecoLst = list('ABDEFGHIJKLMNOPQR')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
for k in range(len(keyLst)):
    dataBox = list()
    key = keyLst[k]
    yRange = rangeLst[k]
    for ss in statLst:
        temp = list()
        for s in ss:
            temp.append(s[key])
        dataBox.append(temp)
    fig = figplot.boxPlot(dataBox, widths=0.5, cLst='rb', label1=ecoLst,
                          label2=None, figsize=(12, 4), yRange=yRange)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'sm_global' + key)
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='rb', label1=ecoLst,
                      label2=['Local', 'Global'], legOnly=True)
saveFile = os.path.join(saveFolder, 'sm_global_legend')
fig.savefig(saveFile)
fig.savefig(saveFile+'.eps')
fig.show()

# significance test
keyLst = ['RMSE', 'Corr']
dfS = pd.DataFrame(index=ecoLst+['All'], columns=keyLst+['N'])
dictA1 = {key: list() for key in keyLst}
dictA2 = {key: list() for key in keyLst}
for k, eco in enumerate(ecoLst):
    for key in keyLst:
        a = statLst[k][0][key]
        b = statLst[k][1][key]
        dictA1[key].append(a)
        dictA2[key].append(b)
        [aa, bb] = utils.rmNan([a, b], returnInd=False)
        s, p = scipy.stats.wilcoxon(aa, bb)
        dfS.at[eco, key] = p
        dfS.at[eco, 'N'] = len(aa)
for key in keyLst:
    a = np.concatenate(dictA1[key])
    b = np.concatenate(dictA2[key])
    [aa, bb] = utils.rmNan([a, b], returnInd=False)
    s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at['All', key] = p
    dfS.at['All', 'N'] = len(aa)
