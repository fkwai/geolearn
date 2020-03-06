import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET

fileCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'codeWQ.csv')
codePdf = pd.read_csv(fileCode, dtype=str).set_index('code')
codeLst = list(codePdf.index)


class DataModelWQ():
    def __init__(self, caseName):
        self.caseName = caseName
        t0 = time.time()
        saveName = os.path.join(kPath.dirWQ, 'trainData', caseName)
        npzFile = np.load(saveName+'.npz')
        self.q = npzFile['q']
        self.f = npzFile['f']
        self.c = npzFile['c']
        self.g = npzFile['g']
        print('loading data {}'.format(time.time()-t0))
        self.info = pd.read_csv(
            saveName+'.csv', index_col=0, dtype={'siteNo': str})
        with open(saveName+'.json', 'r') as fp:
            self.dictData = json.load(fp)
        self.subset = self.loadSubset
        self.dfCount = self.info['siteNo'].value_counts().rename(
            'count').to_frame().rename_axis(index='siteNo')
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
        return cls(caseName)

    def indByRatio(self, ratio):
        # devide training and testing - last 20% as testing
        dfSite = self.dfSite
        ind1 = dfSite[dfSite['pRank'] <= ratio].index.values
        ind2 = dfSite[dfSite['pRank'] > ratio].index.values
        return ind1, ind2

    def indByCount(self, count):
        dfSite = self.dfSite
        indRm = dfSite[dfSite['count'] < count].index.values
        return indRm

    def indByComb(self, codeRm):
        # remove only have obs -> codeRm = ['00010', '00095']
        codeLst = self.dictData['varC']
        indC1 = [codeLst.index(code) for code in codeRm]
        indC2 = [i for i in list(range(len(codeLst))) if i not in indC1]
        temp1 = self.c[:, indC1]
        temp2 = self.c[:, indC2]
        indRm = np.where(~np.isnan(temp1).all(axis=1) &
                         np.isnan(temp2).all(axis=1))[0]
        return indRm

    def saveSubset(self, dictSubsetNew):
        # save to a subset file
        subsetFile = os.path.join(
            kPath.dirWQ, 'trainData', self.caseName+'_subset.json')
        for key, value in dictSubsetNew.items():
            dictSubsetNew[key] = value.tolist()
        if os.path.exists(subsetFile):
            with open(subsetFile, 'r') as fp:
                dictSubset = json.load(fp)
            dictSubset.update(dictSubsetNew)
        else:
            dictSubset = dictSubsetNew
        with open(subsetFile, 'w') as fp:
            json.dump(dictSubset, fp, indent=4)

    def loadSubset(self):
        # save to a subset file
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
    # add a start date to improve efficiency.
    startDate = pd.datetime(1979, 1, 1)

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
            if ctR[0] < startDate:
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
        print('{} on site {} reading {:.3} total {}'.format(
            i, siteNo, t2-t1, t2-t0))
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float64)
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float64)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float64)
    c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float64)
    infoDf = pd.DataFrame(infoLst)

    saveFolder = os.path.join(kPath.dirWQ, 'trainData')
    saveName = os.path.join(saveFolder, caseName)
    np.savez(saveName, q=q, f=f, c=c, g=g)
    infoDf.to_csv(saveName+'.csv')
    dictData = dict(name=caseName, rho=rho, nFill=nFill,
                    varG=varG, varC=varC, siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)
