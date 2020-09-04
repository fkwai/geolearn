import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.app import waterQuality


def wrapData(caseName, siteNoLst, rho=365, freq='D', optC='end'):
    """ wrap up input and target data for the model,as:
    x=[nT,nP,nX]
    y=[nP,nY]
    c=[nP,nC]
    where nP is number of time series
    Arguments:
        caseName {str} -- name of current data case
        siteNoLst {list} -- list of USGS site
    Keyword Arguments:
        rho {int} -- [description] (default: {365})
        nFill {int} -- max number of continous nan to interpolate in input data (default: {5})
        varC {list} -- list of water quality code to learn (default: {usgs.lstCodeSample})
        varG {list} -- list of constant variables in gageII (default: {gageII.lstWaterQuality})
        varQ and varF are fixed so far
    """
    sd = np.datetime64('1979-01-01')
    ed = np.datetime64('2019-12-31')

    # ts data
    varF = gridMET.varLst+ntn.varLst+['distNTN']
    varC = usgs.varC
    varQ = usgs.varQ
    varG = gageII.lstWaterQuality

    # gageII
    tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
    tabG = gageII.updateCode(tabG)

    # read data and merge to: x=[nT,nP,nX], xc=[nP,nY]
    fLst, qLst, cLst, gLst = [list() for x in range(4)]
    infoLst = list()
    t0 = time.time()
    for i, siteNo in enumerate(siteNoLst):
        t1 = time.time()
        varLst = varQ+varC+varF
        df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq=freq)
        dfC = df[varC].dropna(how='all')
        for k in range(len(dfC)):
            ct = dfC.index[k]
            if freq == 'D':
                ctR = pd.date_range(
                    ct-pd.Timedelta(days=rho-1), ct)
            elif freq == 'W':
                ctR = pd.date_range(
                    ct-pd.Timedelta(days=rho*7-1), ct, freq='W-TUE')
            if (ctR[0] < sd) or (ctR[-1] > ed):
                continue
            for lst, var in zip([fLst,  qLst], [varF, varQ]):
                temp = pd.DataFrame({'date': ctR}).set_index(
                    'date').join(df[var])
                # temp = temp.interpolate(
                #     limit=nFill, limit_direction='both', limit_area='inside')
                # give up interpolation after many thoughts
                lst.append(temp.values)
            if optC == 'end':
                cLst.append(dfC.iloc[k].values)
            elif optC == 'seq':
                tempC = pd.DataFrame({'date': ctR}).set_index(
                    'date').join(df[varC])
                cLst.append(tempC.values)
            gLst.append(tabG.loc[siteNo].values)
            infoLst.append(dict(siteNo=siteNo, date=ct))
        t2 = time.time()
        print('{} on site {} reading {:.3f} total {:.3f}'.format(
            i, siteNo, t2-t1, t2-t0))

    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)
    if optC == 'end':
        c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float32)
    elif optC == 'seq':
        c = np.stack(cLst, axis=-1).swapaxes(1, 2).astype(np.float32)

    # save
    infoDf = pd.DataFrame(infoLst)
    saveFolder = os.path.join(kPath.dirWQ, 'trainData')
    saveName = os.path.join(saveFolder, caseName)
    np.savez(saveName, q=q, f=f, c=c, g=g)
    infoDf.to_csv(saveName+'.csv')
    dictData = dict(name=caseName, rho=rho,
                    varG=varG, varC=varC, varQ=['00060'],
                    varF=varF, siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)
