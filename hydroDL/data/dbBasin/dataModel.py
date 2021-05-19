from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL import kPath
import time
import os
import time
import pandas as pd
import numpy as np
import json
from . import io, func
from hydroDL import utils

__all__ = ['DataModelFull', 'DataTrain']


class DataModelFull():
    def __init__(self, caseName):
        self.caseName = caseName
        t0 = time.time()
        saveFolder = io.caseFolder(caseName)
        self.saveFolder = saveFolder
        npz = np.load(os.path.join(saveFolder, 'data.npz'))
        self.q = npz['q']
        self.f = npz['f']
        self.c = npz['c']
        self.g = npz['g']
        with open(os.path.join(saveFolder, 'info')+'.json', 'r') as fp:
            dictData = json.load(fp)
        self.name = dictData['name']
        self.varQ = dictData['varQ']
        self.varF = dictData['varF']
        self.varC = dictData['varC']
        self.varG = dictData['varG']
        self.siteNoLst = dictData['siteNoLst']
        self.sd = np.datetime64(dictData['sd'])
        self.ed = np.datetime64(dictData['ed'])
        self.t = pd.date_range(self.sd, self.ed).values.astype('datetime64[D]')
        self.freq = dictData['freq']
        self.subset = self.loadSubset()
        self.addT()
        self.dataProvision()
        print('loading data {} {:.2f}s'.format(caseName, time.time()-t0))

    @classmethod
    def new(cls, caseName, siteNoLst, nFill=5, freq='D',
            sdStr='1979-01-01', edStr='2019-12-31'):
        print('creating data class')
        io.wrapData(caseName, siteNoLst, nFill=nFill,
                    freq=freq, sdStr=sdStr, edStr=edStr)
        io.initSubset(caseName)
        wqData = cls(caseName)
        return wqData

    def dataProvision(self):
        # set negtive Q to 0
        bq = np.less(self.q, 0., where=~np.isnan(self.q))
        if np.any(bq):
            print('Find negative Q, filled zero')
            self.q[bq] = 0
        self.c = io.nanExt(self.c)  # remove extreme

    def extractT(self, varLst):
        temp = list()
        varTemp = [self.varQ, self.varF, self.varC]
        dataTemp = [self.q, self.f, self.c]
        for var in varLst:
            for dataT, varT in zip(dataTemp, varTemp):
                if var in varT:
                    temp.append(dataT[:, :, varT.index(var)])
        if len(temp) != len(varLst):
            raise Exception('Variable {} not found!'.format(var))
        return (np.stack(temp, axis=2))

    def extractC(self, varLst):
        temp = list()
        varTemp = [self.varG]
        dataTemp = [self.g]
        for var in varLst:
            for dataT, varT in zip(dataTemp, varTemp):
                if var in varT:
                    temp.append(dataT[:, varT.index(var)])
        if len(temp) != len(varLst):
            raise Exception('Variable {} not found!'.format(var))
        return (np.stack(temp, axis=1))

    def saveSubset(self, name, sd=None, ed=None, siteNoLst=None, mask=False):
        subsetFile = os.path.join(self.saveFolder, 'subset.json')
        dictSubset = self.loadSubset()
        if type(mask) is np.ndarray:
            maskFile = os.path.join(
                self.saveFolder, 'mask', name+'.npy')
            np.save(maskFile, mask)
            mask = True
        dictNew = {name: dict(sd=sd, ed=ed, siteNoLst=siteNoLst, mask=mask)}
        dictSubset.update(dictNew)
        with open(subsetFile, 'w') as fp:
            json.dump(dictSubset, fp, indent=4)
        self.subset = self.loadSubset()

    def loadSubset(self, name=None):
        subsetFile = os.path.join(self.saveFolder, 'subset.json')
        if os.path.exists(subsetFile):
            with open(subsetFile, 'r') as fp:
                dictSubset = json.load(fp)
        if name is not None:
            return dictSubset[name]
        else:
            return dictSubset

    def createSubset(self, name, sd=None, ed=None, siteNoLst=None, **kw):
        # create subset by some kws and input - a summary function
        if 'dateLst' in kw:
            tSub = kw['dateLst']
            sdT = self.sd if sd is None else np.datetime64(sd)
            edT = self.ed if ed is None else np.datetime64(ed)
            ns = len(self.siteNoLst) if siteNoLst is None else len(siteNoLst)
            mask = func.createMaskByT(sdT, edT, ns, tSub)
        self.saveSubset(name, sd=sd, ed=ed, siteNoLst=siteNoLst, mask=mask)

    def readSubset(self, name):
        subset = self.loadSubset(name)
        # date
        sdStr = subset['sd']
        edStr = subset['ed']
        sd = self.t[0] if sdStr is None else np.datetime64(sdStr)
        ed = self.t[-1] if edStr is None else np.datetime64(edStr)
        if sd < self.t[0] or ed > self.t[-1]:
            raise Exception('Wrong sd or ed in subset')
        indT1 = np.where(self.t == sd)[0][0]
        indT2 = np.where(self.t == ed)[0][0]+1
        # site
        if subset['siteNoLst'] is None:
            indS = np.arange(len(self.siteNoLst))
        else:
            indS = [self.siteNoLst.index(s) for s in subset['siteNoLst']]
        # mask
        if subset['mask'] is True:
            maskFile = os.path.join(
                self.saveFolder, 'mask', name+'.npy')
            mask = np.load(maskFile)
        else:
            mask = None
        return indT1, indT2, indS, mask

    def extractSubset(self, data, subsetName=None, **kw):
        if subsetName is not None:
            indT1, indT2, indS, mask = self.readSubset(subsetName)
        else:
            indT1, indT2, indS, mask = (
                kw['indT1'], kw['indT2'], kw['indS'], kw['mask'])
        if data is None:
            out = None
        elif data.ndim == 3:
            out = data[indT1:indT2, :, :][:, indS, :]
            if mask is not None:
                out[mask, :] = np.nan
        elif data.ndim == 2:
            out = data[indS, :]
        return out

    def addT(self):
        t = self.t
        ns = len(self.siteNoLst)
        matT, varTLst = io.calT(t)
        matTE = np.repeat(matT[:, None, :], ns, axis=1)
        self.f = np.concatenate([self.f, matTE], axis=2)
        self.varF = self.varF+varTLst


class DataTrain():
    def __init__(self, DM: DataModelFull, **kw):
        dictD = dict(subset='all', varX=DM.varF+DM.varQ,
                     varXC=DM.varG, varY=DM.varC, varYC=None,
                     statX=None, statXC=None, statY=None, statYC=None)
        dictD.update(kw)
        self.caseName = DM.caseName
        # write exclusively, hate those warnings
        self.subset = dictD['subset']
        self.varX = dictD['varX']
        self.varY = dictD['varY']
        self.varXC = dictD['varXC']
        self.varYC = dictD['varYC']
        self.statX = dictD['statX']
        self.statXC = dictD['statXC']
        self.statY = dictD['statY']
        self.statYC = dictD['statYC']

        indT1, indT2, indS, mask = DM.readSubset(self.subset)
        self.t = DM.t[indT1:indT2]
        self.siteNoLst = [DM.siteNoLst[k] for k in indS]
        # upper/lower case for raw/fine data
        X = DM.extractT(self.varX) if self.varX is not None else None
        XC = DM.extractC(self.varXC) if self.varXC is not None else None
        Y = DM.extractT(self.varY) if self.varY is not None else None
        YC = DM.extractC(self.varYC) if self.varYC is not None else None
        kw = dict(indT1=indT1, indT2=indT2, indS=indS, mask=mask)
        self.X = DM.extractSubset(X, **kw)
        self.XC = DM.extractSubset(XC, **kw)
        self.Y = DM.extractSubset(Y, **kw)
        self.YC = DM.extractSubset(YC, **kw)
        self.x, self.statX = self.transIn(
            self.X, self.varX, stat=self.statX)
        self.xc, self.statXC = self.transIn(
            self.XC, self.varXC, stat=self.statXC)
        self.y, self.statY = self.transIn(
            self.Y, self.varY, stat=self.statY)
        self.yc, self.statYC = self.transIn(
            self.YC, self.varYC, stat=self.statYC)

    def tupDataRaw(self):
        return (self.X, self.XC, self.Y, self.YC)

    def tupData(self):
        return (self.x, self.xc, self.y, self.yc)

    def tupVar(self):
        return (self.varX, self.varXC, self.varY, self.varYC)

    def tupStat(self):
        return (self.statX, self.statXC, self.statY, self.statYC)

    def transIn(self, data, var, stat=None):
        mtd = io.extractVarMtd(var)
        # normalize data in
        if data is not None:
            outData, outStat = transform.transIn(data, mtd, stat)
        else:
            (outData, outStat) = (None, None)
        return outData, outStat

    def transOutY(self, data):
        mtd = io.extractVarMtd(self.varY)
        # normalize data out
        if data.shape[-1] == 0:
            out = None
        else:
            out = transform.transOut(data, mtd, self.statY)
        return out
