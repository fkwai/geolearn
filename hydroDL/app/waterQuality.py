import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform

fileCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'codeWQ.csv')
codePdf = pd.read_csv(fileCode, dtype=str).set_index('code')
codeLst = list(codePdf.index)


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
        self.varQ = ['00060']
        self.varF = gridMET.varLst
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
    def new(cls, caseName, siteNoLst, rho=365, nFill=5, varC=codeLst, varG=gageII.lstWaterQuality):
        print('creating data class')
        wrapData(caseName, siteNoLst, rho=rho,
                 nFill=nFill, varC=varC, varG=varG)
        wqData = cls(caseName)
        ind1 = wqData.indByRatio(0.8)
        ind2 = wqData.indByRatio(0.2, first=False)
        wqData.saveSubset(['first80', 'last20'], [ind1, ind2])
        return wqData

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

    def extractSubset(self, subset):
        if subset is not None:
            ind = self.subset[subset]
            q = self.q[:, ind, :]
            c = self.c[ind, :]
            f = self.f[:, ind, :]
            g = self.g[ind, :]
            return (f, g, q, c)
        else:
            return (self.f, self.g, self.q, self.c)

    def transIn(self, subset=None, optQ=1):
        # normalize data in
        if subset is None:
            (f, g, q, c) = (self.f, self.g, self.q, self.c)
        else:
            (f, g, q, c) = self.extractSubset(subset)
        t0 = time.time()
        x, statX = transform.transInAll(
            f, [gridMET.dictStat[var] for var in self.varF])
        t1 = time.time()-t0
        xc, statXC = transform.transInAll(
            g, [gageII.dictStat[var] for var in self.varG])
        t2 = time.time()-t0
        y, statY = transform.transInAll(
            q, [usgs.dictStat[var] for var in self.varQ])
        t3 = time.time()-t0
        yc, statYC = transform.transInAll(
            c, [usgs.dictStat[var] for var in self.varC])
        t4 = time.time()-t0
        print('transform in x->{:.3f} xc->{:.3f} y->{:.3f} yc->{:.3f}'.format(
            t1, t2, t3, t4))
        dataLst, statLst = buildInput(
            (x, xc, y, yc), (statX, statXC, statY, statYC), optQ)
        return dataLst, statLst

    def transOut(self, y, yc, statY, statYC):
        # normalize data out
        t0 = time.time()
        if y.shape[-1] == 0:
            outY = None
        else:
            outY = transform.transOutAll(
                y, [usgs.dictStat[var] for var in self.varQ],  statY)
        t1 = time.time()-t0
        if yc.shape[-1] == 0:
            outYC = None
        else:
            outYC = transform.transOutAll(
                yc, [usgs.dictStat[var] for var in self.varC], statYC)
        t2 = time.time()-t0
        print('transform out y->{:.3f} yc->{:.3f}'.format(t1, t2))
        return outY, outYC

    def calComb(self):
        # calculate the combinations - could improve effeciency later
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

    def calStatC(self, ycP, subset=None):
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


def buildInput(dataLst, statLst, optInput):
    (f, xc, q, yc) = dataLst
    (sF, sXC, sQ, sYC) = statLst
    if optInput == 1:
        x = f
        sX = sF
        y = q
        sY = sQ
    elif optInput == 2:
        x = np.concatenate([q, f], axis=2)
        sX = sQ+sF
        y = None
        sY = None
    elif optInput == 3:
        x = f
        sX = sF
        y = None
        sY = None
    elif optInput == 4:
        x = q
        sX = sQ
        y = None
        sY = None
    return (x, xc, y, yc), (sX, sXC, sY, sYC)


def wrapData(caseName, siteNoLst, rho=365, nFill=5, varC=codeLst, varG=gageII.lstWaterQuality):
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
                    varG=varG, varC=varC, siteNoLst=siteNoLst)
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
