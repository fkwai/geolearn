from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL import kPath
import os
import time
import pandas as pd
import numpy as np
import json
"""
functions for read rawdata and write in caseFolder

[ISSUE] can not deal with water quality for now. 
Need figure out a way to save 3D sparse matrix.
use water temperature for a test
"""
__all__ = ['wrapData', 'readSiteTS', 'extractVarMtd']


def caseFolder(caseName):
    saveFolder = os.path.join(kPath.dirWQ, 'trainDataFull', caseName)
    return saveFolder


def wrapData(caseName, siteNoLst, nFill=5, freq='D',
             sdStr='1979-01-01', edStr='2019-12-31'):
    varF = gridMET.varLst
    varQ = usgs.varQ
    varG = gageII.lstWaterQuality
    varC = ['00010']

    # gageII
    tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
    tabG = gageII.updateCode(tabG)
    tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))
    fLst, qLst, gLst, cLst = [list() for x in range(4)]

    t0 = time.time()
    for i, siteNo in enumerate(siteNoLst):
        t1 = time.time()
        varLst = varQ+varF+varC
        df = readSiteTS(siteNo, varLst=varLst, freq=freq)
        # streamflow
        tempQ = pd.DataFrame({'date': tR}).set_index('date').join(df[varQ])
        qLst.append(tempQ.values)
        # forcings
        tempF = pd.DataFrame({'date': tR}).set_index('date').join(df[varF])
        tempF = tempF.interpolate(
            limit=nFill, limit_direction='both', limit_area='inside')
        fLst.append(tempF.values)
        # # water quality
        tempC = pd.DataFrame({'date': tR}).set_index('date').join(df[varC])
        cLst.append(tempC.values)
        # geog
        gLst.append(tabG.loc[siteNo].values)
        t2 = time.time()
        print('{} on site {} reading {:.3f} total {:.3f}'.format(
            i, siteNo, t2-t1, t2-t0))
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    c = np.stack(cLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)

    # save
    saveFolder = caseFolder(caseName)
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    np.save(os.path.join(saveFolder, 'Q'), q)
    np.save(os.path.join(saveFolder, 'F'), f)
    np.save(os.path.join(saveFolder, 'G'), g)
    np.save(os.path.join(saveFolder, 'C'), c)
    dictData = dict(name=caseName, varG=varG,  varQ=varQ, varF=varF, varC=varC,
                    sd=sdStr, ed=edStr, freq=freq, siteNoLst=siteNoLst)
    with open(os.path.join(saveFolder, 'info')+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)


def readSiteTS(siteNo, varLst, freq='D', area=None,
               sd=np.datetime64('1979-01-01'),
               ed=np.datetime64('2019-12-31')):
    # read data
    td = pd.date_range(sd, ed)
    varC = list(set(varLst).intersection(usgs.varC))
    varQ = list(set(varLst).intersection(usgs.varQ))
    varF = list(set(varLst).intersection(gridMET.varLst))
    varP = list(set(varLst).intersection(ntn.varLst))

    dfD = pd.DataFrame({'date': td}).set_index('date')
    if len(varC) > 0:
        dfC = usgs.readSample(siteNo, codeLst=varC, startDate=sd)
        dfD = dfD.join(dfC)
    if len(varQ) > 0:
        dfQ = usgs.readStreamflow(siteNo, startDate=sd)
        dfQ = dfQ.rename(columns={'00060_00003': '00060'})
        if 'runoff' in varLst:
            if area is None:
                tabArea = gageII.readData(
                    varLst=['DRAIN_SQKM'], siteNoLst=[siteNo])
                area = tabArea['DRAIN_SQKM'].values[0]
            dfQ['runoff'] = calRunoffArea(dfQ['00060'], area)
        dfD = dfD.join(dfQ)
    if len(varF) > 0:
        dfF = gridMET.readBasin(siteNo, varLst=varF)
        dfD = dfD.join(dfF)
    if len(varP) > 0:
        dfP = ntn.readBasin(siteNo, varLst=varP, freq='D')
        dfD = dfD.join(dfP)
    if 'sinT' in varLst or 'cosT' in varLst:
        t = dfD.index.dayofyear.values/365
        dfD['sinT'] = np.sin(2*np.pi*t)
        dfD['cosT'] = np.cos(2*np.pi*t)
    dfD = dfD[varLst]
    if freq == 'D':
        return dfD
    elif freq == 'W':
        dfW = dfD.resample('W-TUE').mean()
        return dfW


def calRunoffArea(q, area):
    # transfer to m/yr
    unitConv = 0.3048**3*365*24*60*60/1000**2
    runoff = q/area*unitConv
    return runoff


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
            else:
                raise Exception('Variable {} not found!'.format(var))
            mtdLst.append(mtd)
    return mtdLst
