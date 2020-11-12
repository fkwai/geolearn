import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath, utils
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import basins
from astropy.timeseries import LombScargle


def calPower(code, df):
    tt = df.index.values
    dfD = df[df[code].notna().values]
    t = dfD.index.values
    x = (t.astype('datetime64[D]') -
         np.datetime64('1979-01-01')).astype(np.float)
    y = dfD[code].values
    nt = len(tt)
    freq = np.fft.fftfreq(nt)[1:]
    ind = np.where((1/freq >= 0) & (1/freq < 1000))[0]
    freq = freq[ind]
    ls = LombScargle(x, y)
    power = ls.power(freq)
    p = ls.false_alarm_probability(power)
    return freq, power, 1-p


class DataModelWQ():
    def __init__(self, caseName, rmFlag=False):
        self.caseName = caseName
        t0 = time.time()
        # data
        saveName = os.path.join(kPath.dirWQ, 'trainData', caseName)
        npzFile = np.load(saveName+'.npz')
        self.q = npzFile['q']
        self.f = npzFile['f']
        self.c = npzFile['c']
        self.g = npzFile['g']
        self.dataProvision()
        if rmFlag is True:
            self.cf = npzFile['cf']
            self.c[self.cf == 1] = np.nan
        print('loading data {}'.format(time.time()-t0))
        # info
        with open(saveName+'.json', 'r') as fp:
            dictData = json.load(fp)

        self.siteNoLst = dictData['siteNoLst']
        self.name = dictData['name']
        self.rho = dictData['rho']
        # self.nFill = dictData['nFill']
        self.varG = dictData['varG']
        self.varC = dictData['varC']
        self.varQ = dictData['varQ']
        self.varF = dictData['varF']
        self.info = pd.read_csv(saveName+'.csv', index_col=0,
                                dtype={'siteNo': str})
        self.info['date'] = self.info['date'].astype('datetime64[D]')
        nWD = len(self.info['date'].dt.dayofweek.unique())
        self.freq = 'W' if nWD == 1 else 'D'
        self.subset = self.loadSubset()
        # self.dfSite = countSite(self.info)
        self.addT()
        print('loading info {}'.format(time.time()-t0))

        if self.q.shape[2] == 1:  # add runoff
            q = self.q[:, :, 0]
            runoff = calRunoff(q, self.info)
            self.q = np.stack([q, runoff], axis=-1).astype(np.float32)
            np.savez(saveName, q=self.q, f=self.f, c=self.c, g=self.g)
            print('calculate Runoff and re-save data {}'.format(time.time()-t0))

    @classmethod
    def new(cls, caseName, siteNoLst, rho=365, freq='D', optC='end'):
        print('creating data class')
        wrapData(caseName, siteNoLst, rho=rho, freq=freq, optC=optC)
        wqData = cls(caseName)
        # ind1 = wqData.indByRatio(0.8)
        # ind2 = wqData.indByRatio(0.8, first=False)
        # wqData.saveSubset(['first80', 'last20'], [ind1, ind2])
        return wqData

    def dataProvision(self):
        bq = np.less(self.q, 0., where=~np.isnan(self.q))
        if np.any(bq):
            print('Find negative Q, filled zero')
            self.q[bq] = 0

    def extractData(self, varTup=None, subset=None):
        dataTup = self.extractVar(varTup)
        dataTup = self.extractSubset(subset, dataTup=dataTup)
        return dataTup

    def extractVar(self, varTup=None):
        if varTup is None:
            varTup = (self.varF, self.varG, self.varQ, self.varC)
        (varX, varXC, varY, varYC) = varTup
        x = self.extractVarT(varX) if varX is not None else None
        xc = self.extractVarC(varXC) if varXC is not None else None
        y = self.extractVarT(varY) if varY is not None else None
        yc = self.extractVarC(varYC) if varYC is not None else None
        return (x, xc, y, yc)

    def extractVarT(self, varLst):
        temp = list()
        for var in varLst:
            if var in self.varQ:
                temp.append(self.q[:, :, 0])
            elif var in self.varF:
                ind = self.varF.index(var)
                temp.append(self.f[:, :, ind])
            elif var in self.varC:  # in
                ind = self.varC.index(var)
                temp.append(self.c[:, :, ind])
            else:
                raise Exception('Variable {} not found!'.format(var))
        return (np.stack(temp, axis=2))

    def extractVarC(self, varLst):
        temp = list()
        for var in varLst:
            if var in self.varC:
                ind = self.varC.index(var)
                temp.append(self.c[:, ind])
            elif var in self.varG:
                ind = self.varG.index(var)
                temp.append(self.g[:, ind])
            else:
                raise Exception('Variable {} not found!'.format(var))
        return (np.stack(temp, axis=1))

    def indByRatio(self, ratio, first=True):
        # devide training and testing - last 20% as testing
        dfSite = self.dfSite
        if first is True:
            ind = dfSite[dfSite['pRank'] <= ratio].index.values
        else:
            ind = dfSite[dfSite['pRank'] > ratio].index.values
        return ind

    def indByCount(self, count):
        dfSite = self.dfSite
        indRm = dfSite[dfSite['count'] < count].index.values
        return indRm

    def indByComb(self, codeRm):
        # remove only have obs -> codeRm = ['00010', '00095']
        codeLst = self.varC
        indC1 = [codeLst.index(code) for code in codeRm]
        indC2 = [i for i in list(range(len(codeLst))) if i not in indC1]
        temp1 = self.c[:, indC1]
        temp2 = self.c[:, indC2]
        indRm = np.where(~np.isnan(temp1).all(axis=1) &
                         np.isnan(temp2).all(axis=1))[0]
        return indRm

    def saveSubset(self, nameLst, indLst):
        if type(nameLst) is not list:
            nameLst = [nameLst]
        if type(indLst) is not list:
            indLst = [indLst]
        dictNew = dict(zip(nameLst, indLst))
        # save to a subset file
        subsetFile = os.path.join(
            kPath.dirWQ, 'trainData', self.caseName+'_subset.json')
        for key, value in dictNew.items():
            dictNew[key] = value.tolist()
        if os.path.exists(subsetFile):
            with open(subsetFile, 'r') as fp:
                dictSubset = json.load(fp)
            dictSubset.update(dictNew)
        else:
            dictSubset = dictNew
        with open(subsetFile, 'w') as fp:
            json.dump(dictSubset, fp, indent=4)
        self.subset = self.loadSubset()

    def loadSubset(self):
        subsetFile = os.path.join(
            kPath.dirWQ, 'trainData', self.caseName+'_subset.json')
        if os.path.exists(subsetFile):
            with open(subsetFile, 'r') as fp:
                dictSubset = json.load(fp)
            for key, value in dictSubset.items():
                dictSubset[key] = np.array(value)
        else:
            dictSubset = None
        return dictSubset

    def extractSubset(self, subset, dataTup=None):
        if dataTup is None:
            dataTup = (self.f, self.g, self.q, self.c)
        if subset is not None:
            ind = self.subset[subset]
            outLst = list()
            for data in dataTup:
                if data is None:
                    out = None
                elif data.ndim == 3:
                    out = data[:, ind, :]
                elif data.ndim == 2:
                    out = data[ind, :]
                outLst.append(out)
            return tuple(outLst)
        else:
            return dataTup

    def subsetInfo(self, subset):
        info = self.info.loc[self.subset[subset].tolist()].reset_index()
        return info

    def transIn(self, statTup=None, subset=None, varTup=None):
        # normalize data in
        dataTup = self.extractData(varTup=varTup, subset=subset)
        t0 = time.time()
        if statTup is None:
            [outDataLst, outStatLst] = [list(), list()]
            for (data, var) in zip(dataTup, varTup):
                if data is not None:
                    mtd = extractVarMtd(var)
                    outData, outStat = transform.transInAll(data, mtd)
                else:
                    (outData, outStat) = (None, None)
                outDataLst.append(outData)
                outStatLst.append(outStat)
            print('transform time {:.3f}'.format(time.time()-t0))
            return outDataLst, outStatLst
        else:
            outDataLst = list()
            for (data, var, stat) in zip(dataTup, varTup, statTup):
                if data is not None:
                    mtd = extractVarMtd(var)
                    outData = transform.transInAll(data, mtd, statLst=stat)
                else:
                    outData = None
                outDataLst.append(outData)
            print('transform time {:.3f}'.format(time.time()-t0))
            return outDataLst

    def transOut(self, data, stat, var):
        mtd = extractVarMtd(var)
        # normalize data out
        t0 = time.time()
        if data.shape[-1] == 0:
            out = None
        else:
            out = transform.transOutAll(
                data, mtd,  stat)
        t1 = time.time()-t0
        print('transform out {}'.format(time.time()-t0))
        return out

    def addT(self):
        nd = pd.DatetimeIndex(self.info['date']).dayofyear.values
        rho = self.rho
        ns = len(nd)
        if self.freq == 'D':
            dd = np.flipud(np.arange(0, rho))
        if self.freq == 'W':
            dd = np.flipud(np.arange(0, rho))*7
        tMat = np.tile(np.expand_dims(nd, axis=1), [1, rho])
        tMat = (tMat-dd)/365
        sinT = np.sin(2*np.pi*tMat).swapaxes(0, 1).astype(np.float32)
        cosT = np.cos(2*np.pi*tMat).swapaxes(0, 1).astype(np.float32)
        matT = np.stack([sinT, cosT], axis=2)
        self.f = np.concatenate([self.f, matT], axis=2)
        self.varF.append('sinT')
        self.varF.append('cosT')

    def calComb(self):
        # calculate the combinations - could improve effeciency later
        codeLst = self.varC
        t0 = time.time()
        dictSum = dict()
        iR, iC = np.where(~np.isnan(self.c))
        aa = np.split(iC, np.cumsum(np.unique(iR, return_counts=True)[1])[:-1])
        aa = [a.tolist() for a in aa]
        for x in set(map(tuple, aa)):
            dictName = '-'.join([codeLst[i] for i in x])
            dictSum[dictName] = aa.count(list(x))
        print('calculate comb {}'.format(time.time()-t0))
        tabComb = pd.DataFrame.from_dict(dictSum, orient='index')
        tabComb = tabComb.sort_values(0, ascending=False)
        return tabComb

    def errBySiteC(self, ycP, varC, subset=None, rmExt=False):
        if type(varC) is not list:
            varC = [varC]
        obsLst = self.extractSubset(subset=subset)
        ycT = obsLst[3]
        indC = [self.varC.index(var) for var in varC]
        info = self.info.loc[self.subset[subset].tolist()].reset_index()
        siteNoLst = self.info.siteNo.unique()
        statMat = np.full([len(siteNoLst), len(indC), 3], np.nan)
        for i, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            for k, iC in enumerate(indC):
                a = ycT[indS, iC]
                b = ycP[indS, k]
                if rmExt is True and len(a) != 0:
                    aV = a[a < np.nanpercentile(a, 95)]
                    aV = aV[aV > np.nanpercentile(a, 5)]
                    ul = np.mean(aV)+np.std(aV)*5
                    a[a > ul] = np.nan
                # indV = np.where(~np.isnan(a))
                if len(indS) > 0:
                    _, indV = utils.rmNan([a, b])
                    rmse = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
                    corr = np.corrcoef(a[indV], b[indV])[0, 1]
                    # nse = 1-np.nansum((b-a)**2)/np.nansum((a-np.nanmean(a))**2)
                    # nse = np.nanmean(b)/np.nanmean(a)-1
                    nse = np.nanmean(np.abs((b-a)/a))
                    statMat[i, k, 0] = rmse
                    statMat[i, k, 1] = corr
                    statMat[i, k, 2] = nse
        return statMat

    def errBySiteQ(self, yP, varQ, subset=None):
        if type(varQ) is not list:
            varQ = [varQ]
        obsLst = self.extractSubset(subset=subset)
        yT = obsLst[2]
        indQ = [self.varQ.index(var) for var in varQ]
        info = self.info.loc[self.subset[subset].tolist()].reset_index()
        siteNoLst = self.info.siteNo.unique()
        statMat = np.full([len(siteNoLst), len(indQ), 2], np.nan)
        for i, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            for k, iQ in enumerate(indQ):
                a = yT[:, indS, iQ]
                b = yP[:, indS, k]
                ns = len(indS)
                rmseMat = np.ndarray(ns)
                corrMat = np.ndarray(ns)
                for kk in range(len(indS)):
                    indV = np.where(~np.isnan(a[:, kk]))
                    rmseMat[kk] = np.sqrt(np.nanmean(
                        (a[indV, kk]-b[indV, kk])**2))
                    corrMat[kk] = np.corrcoef(a[indV, kk], b[indV, kk])[0, 1]
                statMat[i, k, 0] = np.nanmean(rmseMat)
                statMat[i, k, 1] = np.nanmean(corrMat)
        return statMat


def exist(caseName):
    fileName = os.path.join(kPath.dirWQ, 'trainData', caseName)
    if os.path.exists(fileName+'.csv') and \
            os.path.exists(fileName+'.json') and \
            os.path.exists(fileName+'.npz'):
        return True
    else:
        return False


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
    varF = gridMET.varLst+ntn.varLst
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
        df = readSiteTS(siteNo, varLst=varLst, freq=freq)
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
                    varG=varG, varC=varC, varQ=varQ,
                    varF=varF, siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)


def calRunoff(q, info):
    siteNoLst = info.siteNo.unique().tolist()
    dfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
    dfArea.rename({'STAID': 'siteNo'})
    area = info.join(dfArea, on='siteNo')['DRAIN_SQKM'].values
    runoff = calRunoffArea(q, area)
    return runoff


def calRunoffArea(q, area):
    # transfer to m/yr
    unitConv = 0.3048**3*365*24*60*60/1000**2 
    runoff = q/area*unitConv
    return runoff


def countSite(info):
    # counting the dataset used for indexes - fairly fast
    # self.dfCount = self.info['siteNo'].value_counts().rename(
    #     'count').to_frame().rename_axis(index='siteNo')
    dfCount = info['siteNo'].value_counts().rename(
        'count').to_frame().rename_axis('siteNo', axis=0)
    rankSite = info.groupby('siteNo').cumcount().rename('rank')
    dfRank = info.join(rankSite)
    # dfSite = pd.merge(dfRank, dfCount, on='siteNo')
    # removed as will change index number if info is a subset - HF
    dfSite = dfRank.join(dfCount, on='siteNo')
    dfSite['pRank'] = dfSite['rank']/dfSite['count']
    return dfSite


def indYrOddEven(info):
    info['yr'] = pd.DatetimeIndex(info['date']).year
    ind1 = info.index[info['yr'] % 2 == 1].values
    ind2 = info.index[info['yr'] % 2 == 0].values
    indLst = [ind1, ind2]
    return indLst


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
            # add sinT cosT
            elif var in ['sinT', 'cosT']:
                mtd = 'norm'
            else:
                raise Exception('Variable {} not found!'.format(var))
            mtdLst.append(mtd)
    return mtdLst


def calErrSeq(dfP, dfO, tBar=np.datetime64('2000-01-01')):
    sd = np.datetime64('1980-01-01')
    a = dfO[(dfO.index < tBar) & (dfO.index > sd)].values
    b = dfP[(dfP.index < tBar) & (dfP.index > sd)].values
    indV = np.where(~np.isnan(a))
    rmse1 = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
    corr1 = np.corrcoef(a[indV], b[indV])[0, 1]

    a = dfO[dfO.index >= tBar].values
    b = dfP[dfP.index >= tBar].values
    indV = np.where(~np.isnan(a))
    rmse2 = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
    corr2 = np.corrcoef(a[indV], b[indV])[0, 1]
    return [rmse1, rmse2], [corr1, corr2]
