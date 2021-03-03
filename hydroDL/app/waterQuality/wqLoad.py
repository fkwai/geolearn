import numpy as np
import os
import pandas as pd
import json
from hydroDL.master import basins
from hydroDL import kPath, utils
from hydroDL.app import waterQuality

# adhoc codes to simplify current scripts


def loadModel(siteNoLst, outNameLSTM, codeLst):
    # load all sequence
    # LSTM
    dictLSTM = dict()
    for k, siteNo in enumerate(siteNoLst):
        print('\t LSTM site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = basins.loadSeq(outNameLSTM, siteNo)
        dictLSTM[siteNo] = df
    # WRTDS
    dictWRTDS = dict()
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-W', 'B10', 'output')
    for k, siteNo in enumerate(siteNoLst):
        print('\t WRTDS site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirWRTDS, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        # df = utils.time.datePdf(df)
        dictWRTDS[siteNo] = df
    # Observation
    dictObs = dict()
    for k, siteNo in enumerate(siteNoLst):
        print('\t USGS site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = waterQuality.readSiteTS(
            siteNo, varLst=['00060']+codeLst, freq='W', rmFlag=True)
        dictObs[siteNo] = df
    return dictLSTM, dictWRTDS, dictObs,


def dictErr(dictLSTM, dictWRTDS, dictObs, codeLst):
    # calculate correlation
    tt = np.datetime64('2010-01-01')
    t0 = np.datetime64('1980-01-01')
    siteNoLst = list(dictObs.keys())
    # codeLst = dictObs[siteNoLst[0]].columns.tolist()
    t = dictObs[siteNoLst[0]].index.values
    ind1 = np.where((t < tt) & (t >= t0))[0]
    ind2 = np.where(t >= tt)[0]
    corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    rmseMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in siteNoLst:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[ind2].values
            v2 = dictWRTDS[siteNo][code].iloc[ind2].values
            v3 = dictObs[siteNo][code].iloc[ind2].values
            dfQ1 = dictObs[siteNo][['00060', code]].iloc[ind1].dropna()
            (vv1, vv2, vv3), indV = utils.rmNan([v1, v2, v3])
            if (len(indV) < 50) or (len(dfQ1) < 50):
                # print(code, siteNo)
                pass
            else:
                rmse1, corr1 = utils.stat.calErr(vv1, vv2)
                rmse2, corr2 = utils.stat.calErr(vv1, vv3)
                rmse3, corr3 = utils.stat.calErr(vv2, vv3)
                corrMat[indS, ic, 0] = corr1
                corrMat[indS, ic, 1] = corr2
                corrMat[indS, ic, 2] = corr3
                rmseMat[indS, ic, 0] = rmse1
                rmseMat[indS, ic, 1] = rmse2
                rmseMat[indS, ic, 2] = rmse3
    return corrMat, rmseMat
