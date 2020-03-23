import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform


class DataModelWQ():
    def __init__(self, caseName):
        self.caseName = caseName
        t0 = time.time()
        # data
        saveName = os.path.join(kPath.dirWQ, 'trainData', caseName)
        npzFile = np.load(saveName+'.npz')
        self.q = npzFile['q']
        self.f = npzFile['f']
        self.c = npzFile['c']
        self.g = npzFile['g']
        print('loading data {}'.format(time.time()-t0))
        # info
        with open(saveName+'.json', 'r') as fp:
            dictData = json.load(fp)
        self.siteNoLst = dictData['siteNoLst']
        self.name = dictData['name']
        self.rho = dictData['rho']
        self.nFill = dictData['nFill']
        self.varG = dictData['varG']
        self.varC = dictData['varC']
        self.varQ = ['00060']  # delete later
        self.varF = gridMET.varLst  # delete later
        self.info = pd.read_csv(
            saveName+'.csv', index_col=0, dtype={'siteNo': str})
        self.subset = self.loadSubset()
        # counting the dataset used for indexes - fairly fast
        # self.dfCount = self.info['siteNo'].value_counts().rename(
        #     'count').to_frame().rename_axis(index='siteNo')
        self.dfCount = self.info['siteNo'].value_counts().rename(
            'count').to_frame().rename_axis('siteNo', axis=0)

        rankSite = self.info.groupby('siteNo').cumcount().rename('rank')
        dfRank = self.info.join(rankSite)
        dfSite = pd.merge(dfRank, self.dfCount, on='siteNo')
        dfSite['pRank'] = dfSite['rank']/dfSite['count']
        self.dfSite = dfSite
        print('loading info {}'.format(time.time()-t0))

    @classmethod
    def new(cls, caseName, siteNoLst, rho=365, nFill=5, varC=usgs.varC, varG=gageII.lstWaterQuality):
        print('creating data class')
        wrapData(caseName, siteNoLst, rho=rho,
                 nFill=nFill, varC=varC, varG=varG)
        wqData = cls(caseName)
        ind1 = wqData.indByRatio(0.8)
        ind2 = wqData.indByRatio(0.8, first=False)
        wqData.saveSubset(['first80', 'last20'], [ind1, ind2])
        return wqData

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

    def extractVarMtd(self, varLst):
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
                mtdLst.append(mtd)
        return mtdLst

    def extractVarT(self, varLst):
        temp = list()
        for var in varLst:
            if var in self.varQ:
                temp.append(self.q[:, :, 0])
            elif var in self.varF:
                ind = self.varF.index(var)
                temp.append(self.f[:, :, ind])
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
        self.loadSubset()

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
                    mtd = self.extractVarMtd(var)
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
                    mtd = self.extractVarMtd(var)
                    outData = transform.transInAll(data, mtd, statLst=stat)
                else:
                    outData = None
                outDataLst.append(outData)
            print('transform time {:.3f}'.format(time.time()-t0))
            return outDataLst

    def transOut(self, data, stat, var):
        mtd = self.extractVarMtd(var)
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

    def errBySite(self, ycP, subset=None):
        obsLst = self.extractSubset(subset=subset)
        ycT = obsLst[3]
        info = self.info.loc[self.subset[subset].tolist()].reset_index()
        siteNoLst = self.info.siteNo.unique()
        nc = ycT.shape[-1]
        statMat = np.full([len(siteNoLst), nc, 2], np.nan)
        for i, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            for k in range(nc):
                a = ycT[indS, k]
                b = ycP[indS, k]
                indV = np.where(~np.isnan(a))
                rmse = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
                corr = np.corrcoef(a[indV], b[indV])[0, 1]
                statMat[i, k, 0] = rmse
                statMat[i, k, 1] = corr
        return statMat


def exist(caseName):
    fileName = os.path.join(kPath.dirWQ, 'trainData', caseName)
    if os.path.exists(fileName+'.csv') and \
            os.path.exists(fileName+'.json') and \
            os.path.exists(fileName+'.npz'):
        return True
    else:
        return False


def wrapData(caseName, siteNoLst, rho=365, nFill=5, varC=usgs.varC, varG=gageII.lstWaterQuality):
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
    # add a start/end date to improve efficiency.
    startDate = pd.datetime(1979, 1, 1)
    endDate = pd.datetime(2019, 12, 31)

    # gageII
    tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
    tabG = gageII.updateCode(tabG)

    # read data and merge to: f/q=[nT,nP,nX], g/c=[nP,nY]
    fLst = list()  # forcing ts
    gLst = list()  # geo-const
    qLst = list()  # streamflow
    cLst = list()  # water quality
    infoLst = list()
    t0 = time.time()
    for i, siteNo in enumerate(siteNoLst):
        t1 = time.time()
        dfC = usgs.readSample(siteNo, codeLst=varC, startDate=startDate)
        dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
        dfF = gridMET.readBasin(siteNo)
        for k in range(len(dfC)):
            ct = dfC.index[k]
            ctR = pd.date_range(ct-pd.Timedelta(days=rho-1), ct)
            if (ctR[0] < startDate) or (ctR[-1] > endDate):
                continue
            tempQ = pd.DataFrame({'date': ctR}).set_index('date').join(
                dfQ).interpolate(limit=nFill, limit_direction='both')
            tempF = pd.DataFrame({'date': ctR}).set_index('date').join(
                dfF).interpolate(limit=nFill, limit_direction='both')
            qLst.append(tempQ.values)
            fLst.append(tempF.values)
            cLst.append(dfC.iloc[k].values)
            gLst.append(tabG.loc[siteNo].values)
            infoLst.append(dict(siteNo=siteNo, date=ct))
        t2 = time.time()
        print('{} on site {} reading {:.3f} total {:.3f}'.format(
            i, siteNo, t2-t1, t2-t0))
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)
    c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float32)
    infoDf = pd.DataFrame(infoLst)

    saveFolder = os.path.join(kPath.dirWQ, 'trainData')
    saveName = os.path.join(saveFolder, caseName)
    np.savez(saveName, q=q, f=f, c=c, g=g)
    infoDf.to_csv(saveName+'.csv')
    dictData = dict(name=caseName, rho=rho, nFill=nFill,
                    varG=varG, varC=varC, varQ=['00060'], varF=gridMET.varLst,
                    siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)

# find the distribution of data
# for k, var in enumerate(wqData.varC):
#     fig, axes = plt.subplots(2, 2)
#     temp = wqData.c[:, k].flatten()
#     temp90 = temp[np.where((temp > np.nanpercentile(temp, 5)) &
#                            (temp < np.nanpercentile(temp, 95)))]
#     axes[0, 0].hist(temp, bins=100)
#     axes[0, 1].hist(temp90, bins=100)
#     try:
#         axes[1, 0].hist(np.log(temp+1), bins=100)
#         axes[1, 1].hist(np.log(temp90+1), bins=100)
#     except(ValueError):
#         print(var+' can not log')
#     fig.suptitle(var)
#     fig.show()
