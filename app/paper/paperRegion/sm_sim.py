import scipy
import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL import master
from hydroDL.post import figplot, stat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
import matplotlib
import pandas as pd

caseLst = ['080305', '090301', '090303',
           '090401', '090402', '100105', '100204']
caseLabLst = ['8.3.5', '9.3.1', '9.3.3',
              '9.4.1', '9.4.2', '10.1.5', '10.2.4']
saveFolder = r'C:\Users\geofk\work\paper\SMAP-regional'
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# test
tRange = [20160401, 20180401]
subsetPattern = 'ecoReg_{}_L{}_v2f1'
levLst = [3, 2, 1, 0]
rootDB = pathSMAP['DB_L3_NA']
dfC = dbCsv.DataframeCsv(rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
latC, lonC = dfC.getGeo()

errLstAll = list()
for case in caseLst:
    testName = subsetPattern.format(case, 3)
    errLst = list()
    for k in levLst:
        if k in [0, 1]:
            subset = 'ecoReg_{}_L{}_v2f1'.format(case, k)
        else:
            subset = subsetPattern.format(case, k)
        outName = subset + '_Forcing'
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionCase', outName)
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        err = stat.statError(yp[:, :, 0], yt[:, :, 0])
        errLst.append(err)
    errLstAll.append(errLst)

# plot box
cLst = 'ygbr'
keyLst = ['RMSE', 'Corr']
for key in keyLst:
    dataBox = list()
    for errLst in errLstAll:
        temp = list()
        for err in errLst:
            temp.append(err[key])
        dataBox.append(temp)
    fig = figplot.boxPlot(dataBox, label1=caseLabLst, cLst=cLst,
                          figsize=(12, 4), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'sm_sim_{}'.format(key))
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')

label2 = ['local', 'local+close', 'local+far', 'local+dissimilar']
fig = figplot.boxPlot(dataBox, label2=label2, cLst=cLst,  legOnly=True)
saveFile = os.path.join(saveFolder, 'sm_sim_legend')
fig.savefig(saveFile)
fig.savefig(saveFile+'.eps')
fig.show()


# significance test
keyLst = ['RMSE', 'Corr']
columns = ['local vs near',
           'near vs far',
           'far vs dissimilar',
           'near vs dissimilar']
dfS = pd.DataFrame(index=caseLabLst+['All'], columns=columns+['N'])
key = 'RMSE'
aLst, bLst, cLst, dLst = [list() for k in range(4)]
for k, eco in enumerate(caseLabLst):
    a = errLstAll[k][0][key]
    b = errLstAll[k][1][key]
    c = errLstAll[k][2][key]
    d = errLstAll[k][3][key]
    s, p = scipy.stats.wilcoxon(a, b)
    dfS.at[eco, 'local vs near'] = p
    s, p = scipy.stats.wilcoxon(b, c)
    dfS.at[eco, 'near vs far'] = p
    s, p = scipy.stats.wilcoxon(c, d)
    dfS.at[eco, 'far vs dissimilar'] = p
    s, p = scipy.stats.wilcoxon(b, d)
    dfS.at[eco, 'near vs dissimilar'] = p
    dfS.at[eco, 'N'] = len(a)
    aLst.append(a)
    bLst.append(b)
    cLst.append(c)
    dLst.append(d)

a = np.concatenate(aLst)
b = np.concatenate(bLst)
c = np.concatenate(cLst)
d = np.concatenate(dLst)
s, p = scipy.stats.wilcoxon(a, b)
dfS.at['All', 'local vs near'] = p
s, p = scipy.stats.wilcoxon(b, c)
dfS.at['All', 'near vs far'] = p
s, p = scipy.stats.wilcoxon(c, d)
dfS.at['All', 'far vs dissimilar'] = p
s, p = scipy.stats.wilcoxon(b, d)
dfS.at['All', 'near vs dissimilar'] = p
dfS.at['All', 'N'] = len(a)
