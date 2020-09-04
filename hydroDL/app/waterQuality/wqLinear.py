from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, transform

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
from sklearn.linear_model import LinearRegression


def loadSeq(siteNo, varY, model, optX='F', optT='Y8090', order=(5, 0, 5)):
    if model == 'ARMA':
        dirAR = os.path.join(kPath.dirWQ, 'modelStat', 'ARMA')
        strOrder = '-'.join([str(k) for k in order])
        saveFolderName = '{}-{}-{}-{}'.format(optX, optT, varY, strOrder)
        saveFolder = os.path.join(dirAR, saveFolderName)
    elif model == 'LR':
        dirLR = os.path.join(kPath.dirWQ, 'modelStat', 'LR')
        saveFolderName = '{}-{}-{}'.format(optX, optT, varY)
        saveFolder = os.path.join(dirLR, saveFolderName)
    else:
        raise Exception('model {} invalid!'.format(model))
    predFile = os.path.join(saveFolder, siteNo)
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)

    if os.path.exists(predFile):
        dfP = pd.read_csv(predFile, index_col=None)
        dfP = utils.time.datePdf(dfP)
    else:
        if optX == 'F':
            varX = gridMET.varLst
        elif optX == 'QF':
            varX = ['00060'] + gridMET.varLst
        else:
            raise Exception('optX {} invalid!'.format(optX))
        dfX = waterQuality.readSiteX(siteNo, varX)
        dfY = waterQuality.readSiteY(siteNo, [varY])
        # normalize
        mtdX = waterQuality.extractVarMtd(varX)
        normX, statX = transform.transInAll(dfX.values, mtdX)
        dfXN = pd.DataFrame(data=normX, index=dfX.index, columns=dfX.columns)
        mtdY = waterQuality.extractVarMtd([varY])
        normY, statY = transform.transInAll(dfY.values, mtdY)
        dfYN = pd.DataFrame(data=normY, index=dfY.index, columns=dfY.columns)
        if optT == 'Y8090':
            dfXT = dfXN[dfXN.index < np.datetime64('2000-01-01')]
            dfYT = dfYN[dfYN.index < np.datetime64('2000-01-01')]
        elif optT == 'Y0010':
            dfXT = dfXN[dfXN.index >= np.datetime64('2000-01-01')]
            dfYT = dfYN[dfYN.index >= np.datetime64('2000-01-01')]
        else:
            raise Exception('optT {} invalid!'.format(optT))

        # train and test
        if model == 'ARMA':
            dfPN, resT = trainARMA(dfXT, dfYT, dfXN, dfYN, order)
        if model == 'LR':
            dfPN = trainLR(dfXT, dfYT, dfXN, dfYN)
        yP = transform.transOut(dfPN.values, mtdY[0], statY[0])
        dfP = pd.DataFrame(data=yP, index=dfYN.index, columns=dfYN.columns)

        # save result, model, stat
        dfP.reset_index().to_csv(predFile, index=False)
        statFile = os.path.join(saveFolder, siteNo+'_stat.json')
        with open(statFile, 'w') as fp:
            json.dump(dict(statX=statX, statY=statY), fp, indent=4)
        # save model
        # if model == 'ARMA':
        #     modelFile = os.path.join(saveFolder, siteNo+'_model.p')
        #     resT.save(modelFile)
    return dfP


def trainARMA(dfXT, dfYT, dfXN, dfYN, order):
    # train model
    try:
        modT = sm.tsa.statespace.SARIMAX(dfYT, exog=dfXT, order=order)
        resT = modT.fit(disp=False)
    except:
        print('try approximate_diffuse init')
        modT = sm.tsa.statespace.SARIMAX(
            dfYT, exog=dfXT, order=order, initialization='approximate_diffuse')
        dfYT = dfYT.replace([np.inf, -np.inf], np.nan)
        resT = modT.fit(disp=False)
    # test model
    mod = sm.tsa.statespace.SARIMAX(dfYN, exog=dfXN, order=order)
    res = mod.filter(resT.params)
    pred = res.get_prediction()
    dfPN = pred.predicted_mean
    return dfPN, resT


def trainLR(dfXT, dfYT, dfXN, dfYN):
    [xx, yy], iv = utils.rmNan([dfXT.values, dfYT.values])
    if len(iv) > 0:
        modelYC = LinearRegression().fit(xx, yy)
        yp = modelYC.predict(dfXN.values)
        dfPN = pd.DataFrame(data=yp, index=dfYN.index, columns=dfYN.columns)
    else:
        dfPN = pd.DataFrame(
            index=dfYN.index, columns=dfYN.columns, data=np.nan)
    return dfPN
