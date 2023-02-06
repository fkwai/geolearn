from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, camels
from hydroDL import kPath, utils
import os
import time
import pandas as pd
import numpy as np
import json

"""
functions for read rawdata and write in caseFolder

"""
__all__ = ['wrapData', 'readSiteTS', 'extractVarMtd', 'label2var']

varTLst = ['datenum', 'sinT', 'cosT']


def caseFolder(caseName):
    saveFolder = os.path.join(kPath.dirWQ, 'trainDataFull', caseName)
    return saveFolder


def wrapData(
    caseName,
    siteNoLst,
    nFill=5,
    freq='D',
    sdStr='1979-01-01',
    edStr='2019-12-31',
    varF=gridMET.varLst + ntn.varLst + GLASS.varLst,
    varQ=usgs.varQ,
    varG=gageII.varLst,
    varC=usgs.varC,
    rmFlag='True',
):
    # gageII
    tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
    tabG = gageII.updateCode(tabG)
    sd = np.datetime64(sdStr)
    ed = np.datetime64(edStr)
    tR = pd.date_range(sd, ed)
    fLst, qLst, gLst, cLst = [list() for x in range(4)]

    t0 = time.time()
    for i, siteNo in enumerate(siteNoLst):
        t1 = time.time()
        varLst = varQ + varF + varC
        df = readSiteTS(
            siteNo, varLst=varLst, freq=freq, rmFlag=rmFlag, sd=sd, ed=ed
        )
        # streamflow
        tempQ = pd.DataFrame({'date': tR}).set_index('date').join(df[varQ])
        qLst.append(tempQ.values)
        # forcings
        tempF = pd.DataFrame({'date': tR}).set_index('date').join(df[varF])
        tempF = tempF.interpolate(
            limit=nFill, limit_direction='both', limit_area='inside'
        )
        fLst.append(tempF.values)
        # # water quality
        tempC = pd.DataFrame({'date': tR}).set_index('date').join(df[varC])
        cLst.append(tempC.values)
        # geog
        gLst.append(tabG.loc[siteNo].values)
        t2 = time.time()
        print(
            '{} site {} reading {:.3f} total {:.3f}'.format(i, siteNo, t2 - t1, t2 - t0)
        )
    print('wrap F')
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    print('wrap Q')
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    print('wrap C')
    c = np.stack(cLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)
    print('save')
    # save
    saveDataFrame(
        caseName,
        c=c,
        q=q,
        f=f,
        g=g,
        varC=varC,
        varQ=varQ,
        varF=varF,
        varG=varG,
        sdStr=sdStr,
        edStr=edStr,
        freq=freq,
        siteNoLst=siteNoLst,
    )


def wrapDataCamels(
    caseName,
    siteNoLst,
    nFill=5,
    freq='D',
    sdStr='1980-01-01',
    edStr='2014-12-31',
    optF='nldas',
    varF=camels.varF,
    varQ=camels.varQ,
    varG=camels.varG,
):
    tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))
    dfG = camels.readAttr(siteNoLst=siteNoLst, varLst=camels.varG)
    fLst, qLst = [list() for x in range(2)]
    siteNo = siteNoLst[0]
    for i, siteNo in enumerate(siteNoLst):
        dfQ = camels.readStreamflow(siteNo)
        dfF = camels.readForcing(siteNo, opt=optF)
        tempQ = pd.DataFrame({'date': tR}).set_index('date').join(dfQ[varQ])
        qLst.append(tempQ.values)
        tempF = pd.DataFrame({'date': tR}).set_index('date').join(dfF[varF])
        tempF = tempF.interpolate(
            limit=nFill, limit_direction='both', limit_area='inside'
        )
        fLst.append(tempF.values)
        print('{} on site {}'.format(i, siteNo))
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    g = dfG.values.astype(np.float32)
    c = np.ndarray([len(tR), len(siteNoLst), 0])  # adhoc
    varC = []  # adhoc
    # save
    saveDataFrame(
        caseName,
        c=c,
        q=q,
        f=f,
        g=g,
        varC=varC,
        varQ=varQ,
        varF=varF,
        varG=varG,
        sdStr=sdStr,
        edStr=edStr,
        freq=freq,
        siteNoLst=siteNoLst,
    )
    initSubset(caseName)


def saveDataFrame(
    caseName, *, c, q, f, g, varC, varQ, varF, varG, sdStr, edStr, freq, siteNoLst
):
    # save
    saveFolder = caseFolder(caseName)
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    np.savez_compressed(os.path.join(saveFolder, 'data'), c=c, q=q, f=f, g=g)
    dictData = dict(
        name=caseName,
        varC=varC,
        varQ=varQ,
        varF=varF,
        varG=varG,
        sd=sdStr,
        ed=edStr,
        freq=freq,
        siteNoLst=siteNoLst,
    )
    with open(os.path.join(saveFolder, 'info') + '.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)


def initSubset(caseName):
    saveFolder = caseFolder(caseName)
    subsetFile = os.path.join(saveFolder, 'subset.json')
    dictSubset = dict(all=dict(sd=None, ed=None, siteNoLst=None, mask=False))
    with open(subsetFile, 'w') as fp:
        json.dump(dictSubset, fp, indent=4)
    maskFolder = os.path.join(saveFolder, 'mask')
    if not os.path.exists(maskFolder):
        os.mkdir(maskFolder)


def readSiteTS(
    siteNo,
    varLst,
    freq='D',
    area=None,
    sd=np.datetime64('1979-01-01'),
    ed=np.datetime64('2019-12-31'),
    rmFlag=True,
):
    # read data
    td = pd.date_range(sd, ed)
    varC = list(set(varLst).intersection(usgs.sampleFull))
    varQ = list(set(varLst).intersection(usgs.varQ))
    varF = list(set(varLst).intersection(gridMET.varLst))
    varP = list(set(varLst).intersection(ntn.varLst))
    varR = list(set(varLst).intersection(GLASS.varLst))
    varT = list(set(varLst).intersection(varTLst))

    dfD = pd.DataFrame({'date': td}).set_index('date')
    if len(varC) > 0:
        if rmFlag:
            dfC, dfCF = usgs.readSample(
                siteNo, codeLst=varC, startDate=sd, flag=2, csv=True
            )
            if dfC is not None:
                dfC = usgs.removeFlag(dfC, dfCF)
        else:
            dfC = usgs.readSample(siteNo, codeLst=varC, startDate=sd, csv=True)
        if dfC is None:
            dfC = pd.DataFrame(index=td, columns=varC)
        dfD = dfD.join(dfC)
    if len(varQ) > 0:
        dfQ = usgs.readStreamflow(siteNo, startDate=sd)
        if dfQ is None:
            dfQ = pd.DataFrame(index=td, columns=varQ)
        else:
            dfQ = dfQ.rename(columns={'00060_00003': '00060'})
            if 'runoff' in varLst:
                if area is None:
                    tabArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=[siteNo])
                    area = tabArea['DRAIN_SQKM'].values[0]
                dfQ['runoff'] = calRunoffArea(dfQ['00060'], area)
        dfD = dfD.join(dfQ)
    if len(varF) > 0:
        dfF = gridMET.readBasin(siteNo, varLst=varF)
        if 'tmmx' in varF:
            dfF['tmmx'] = dfF['tmmx'] - 273.15
        if 'tmmn' in varF:
            dfF['tmmn'] = dfF['tmmn'] - 273.15
        dfD = dfD.join(dfF)
    if len(varP) > 0:
        dfP = ntn.readBasin(siteNo, varLst=varP, freq='D')
        dfD = dfD.join(dfP)
    if len(varR) > 0:
        dfR = GLASS.readBasin(siteNo, varLst=varR, freq='D')
        dfD = dfD.join(dfR)
    if len(varT) > 0:
        t = dfD.index.values
        matT, _ = calT(t)
        dfT = pd.DataFrame(index=t, columns=varTLst, data=matT)
        dfD = dfD.join(dfT[varT])
    dfD = dfD[varLst]
    if freq == 'D':
        return dfD
    elif freq == 'W':
        dfW = dfD.resample('W-TUE').mean()
        return dfW


def calRunoffArea(q, area):
    # transfer q[ft^3/s] area [sqkm] to mm/day
    unitConv = 0.3048**3 * 24 * 60 * 60 / 1000
    runoff = q / area * unitConv
    return runoff


def calT(t):
    # t of datetime64[D]
    tn = utils.time.date2num(t)
    sinT = np.sin(2 * np.pi * tn / 365.24)
    cosT = np.cos(2 * np.pi * tn / 365.24)
    matT = np.stack([tn, sinT, cosT], axis=1)
    return matT, varTLst


def extractVarMtd(varLst):
    mtdLst = list()
    if varLst is None:
        mtdLst = None
    else:
        for var in varLst:
            if var in gridMET.dictStat.keys():
                mtd = gridMET.dictStat[var]
            elif var in gageII.dictStat.keys():
                mtd = gageII.dictStat[var]
            elif var in usgs.dictStat.keys():
                mtd = usgs.dictStat[var]
            elif var in ntn.dictStat.keys():
                mtd = ntn.dictStat[var]
            elif var in GLASS.dictStat.keys():
                mtd = GLASS.dictStat[var]
            elif var in ['datenum', 'sinT', 'cosT']:
                mtd = 'norm'
            else:
                raise Exception('Variable {} not found!'.format(var))
            mtdLst.append(mtd)
    return mtdLst


def label2var(label, norm=False):
    dictVar = dict(
        F=gridMET.varLst,
        Q=['00060'],
        P=ntn.varLst,
        T=['datenum', 'sinT', 'cosT'],
        R=GLASS.varLst,
        C=usgs.varC,
    )
    if norm is True:
        dictVar['C'] = [c + '-N' for c in usgs.newC]
    varLst = list()
    for x in label:
        varLst = varLst + dictVar[x]
    return varLst


def nanPerc(data, p=5):
    v1 = np.nanpercentile(data, p, axis=0)
    v2 = np.nanpercentile(data, 100 - p)
    out = data.copy()
    out[out < v1] = np.nan
    out[out > v2] = np.nan
    return out


def nanExt(data, p=10, n=5):
    out = data.copy()
    v1 = np.nanpercentile(data, p, axis=0)
    v2 = np.nanpercentile(data, 100 - p, axis=0)
    vr = v2 - v1
    b1 = data > (v2 + vr * n)
    b2 = data < (v1 - vr * n)
    out[b1] = np.nan
    out[b2] = np.nan
    print('{} extremes removed'.format(np.sum(b1) + np.sum(b2)))
    return out
