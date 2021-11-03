import pickle
from hydroDL.data import Dataframe, DataModel
from hydroDL.data import transform, gageII, usgs,  gridMET, ntn, GLASS
from hydroDL import kPath
import time
import os
import time
import pandas as pd
import numpy as np
import json
from . import io, func
from hydroDL import utils
from .io import *
from .func import *
from sklearn.preprocessing import QuantileTransformer, PowerTransformer


class DataFrameBasin():
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
            sdStr='1979-01-01', edStr='2019-12-31',
            varF=gridMET.varLst+ntn.varLst+GLASS.varLst,
            varQ=usgs.varQ, varG=gageII.varLst, varC=usgs.newC):
        print('creating data class')
        siteNoLst.sort()
        io.wrapData(caseName, siteNoLst, nFill=nFill,
                    freq=freq, sdStr=sdStr, edStr=edStr,
                    varF=varF, varQ=varQ, varG=varG, varC=varC)
        io.initSubset(caseName)
        wqData = cls(caseName)
        return wqData

    def saveAs(self, caseName):
        io.saveDataFrame(caseName, c=self.c, q=self.q, f=self.f, g=self.g,
                         varC=self.varC, varQ=self.varQ, varF=self.varF, varG=self.varG,
                         sdStr=str(self.sd), edStr=str(self.ed), freq=self.freq,
                         siteNoLst=self.siteNoLst)
        # save subset
        saveFolder = io.caseFolder(caseName)
        subsetFile = os.path.join(saveFolder, 'subset.json')
        with open(subsetFile, 'w') as fp:
            json.dump(self.subset, fp, indent=4)

    def getT(self, subsetName=None):
        if subsetName is None:
            return self.t
        else:
            indT1, indT2, _, _ = self.readSubset(subsetName)
            return self.t[indT1:indT2]

    def getSite(self, subsetName=None):
        if subsetName is None:
            return self.siteNoLst
        else:
            _, _, indS, _ = self.readSubset(subsetName)
            return [self.siteNoLst[k] for k in indS]

    def getGeo(self, subsetName=None):
        siteNoLst = self.getSite(subsetName=subsetName)
        dfCrd = gageII.readData(
            varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
        lat = dfCrd['LAT_GAGE'].values
        lon = dfCrd['LNG_GAGE'].values
        return lat, lon

    def dataProvision(self):
        # set negtive Q to 0
        bq = np.less(self.q, 0., where=~np.isnan(self.q))
        if np.any(bq):
            print('Find negative Q, filled zero')
            self.q[bq] = 0
        self.c = io.nanExt(self.c, p=10, n=5)  # remove extreme

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

    def createSubset(self, name, sd=None, ed=None, siteNoLst=None, mask=False, **kw):
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
            mask = False
        # out = np.
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
            if mask is not False:
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


class DataModelBasin(DataModel):
    def __init__(self, dataFrame, **kw):
        super(DataModelBasin, self).__init__()
        if type(dataFrame) is str:
            DF = DataFrameBasin(dataFrame)
        elif utils.sameClass(type(dataFrame), DataFrameBasin):
            DF = dataFrame
        else:
            raise Exception('unknown DataFrame')
        dictD = dict(subset='all', varX=DF.varF+DF.varQ,
                     varXC=DF.varG, varY=DF.varC, varYC=None)
        dictD.update(kw)
        self.caseName = DF.caseName
        self.dictD = dictD
        # write exclusively, hate those warnings
        self.strLst = ['X', 'XC', 'Y', 'YC']
        self.subset = dictD['subset']
        self.varX = dictD['varX']
        self.varY = dictD['varY']
        self.varXC = dictD['varXC']
        self.varYC = dictD['varYC']

        indT1, indT2, indS, mask = DF.readSubset(self.subset)
        self.t = DF.t[indT1:indT2]
        self.siteNoLst = [DF.siteNoLst[k] for k in indS]
        # upper/lower case for raw/fine data
        X = DF.extractT(self.varX) if self.varX is not None else None
        XC = DF.extractC(self.varXC) if self.varXC is not None else None
        Y = DF.extractT(self.varY) if self.varY is not None else None
        YC = DF.extractC(self.varYC) if self.varYC is not None else None
        kwX = dict(indT1=indT1, indT2=indT2, indS=indS, mask=False)
        self.X = DF.extractSubset(X, **kwX)
        self.XC = DF.extractSubset(XC, **kwX)
        kwY = dict(indT1=indT1, indT2=indT2, indS=indS, mask=mask)
        self.Y = DF.extractSubset(Y, **kwY)
        self.YC = DF.extractSubset(YC, **kwY)

    def trans(self, *, mtdX=None, mtdXC=None, mtdY=None, mtdYC=None):
        mtdLst = [mtdX, mtdXC, mtdY, mtdYC]
        varLst = [self.varX, self.varXC, self.varY, self.varYC]
        statLst = list()
        for mtd, var in zip(mtdLst, varLst):
            if mtd is None:
                # mtd = io.extractVarMtd(var)
                mtd = None if var is None else ['skip' for x in var]
            statLst.append(dict(mtdLst=mtd))
        self.transIn(statX=statLst[0], statXC=statLst[1],
                     statY=statLst[2], statYC=statLst[3])

    def transIn(self, *, statX, statXC, statY, statYC):
        if self.X is not None:
            self.x, self.statX = transform.transIn(self.X, **statX)
        if self.XC is not None:
            self.xc, self.statXC = transform.transIn(self.XC, **statXC)
        if self.Y is not None:
            self.y, self.statY = transform.transIn(self.Y, **statY)
        if self.YC is not None:
            self.yc, self.statYC = transform.transIn(self.YC, **statYC)

    def borrowStat(self, DM: DataModel):
        self.transIn(statX=DM.statX, statXC=DM.statXC,
                     statY=DM.statY, statYC=DM.statYC)

    def getDataRaw(self):
        return self.X, self.XC, self.Y, self.YC

    def save(self):
        pass

    def load(self):
        pass

    def saveStat(self, saveFolder):
        for s in self.strLst:
            stat = getattr(self, 'stat'+s)
            statFile = os.path.join(saveFolder, 'stat'+s)
            with open(statFile, 'wb') as f:
                pickle.dump(stat, f)

    def loadStat(self, saveFolder):
        for s in self.strLst:
            dataRaw = getattr(self, s)
            if dataRaw is not None:
                statFile = os.path.join(saveFolder, 'stat'+s)
                with open(statFile, 'rb') as f:
                    stat = pickle.load(f)
                    setattr(self, 'stat'+s, stat)
        self.transIn(statX=self.statX, statXC=self.statXC,
                     statY=self.statY, statYC=self.statYC)

    def transOutY(self, yP):
        if yP.shape[-1] != 0:
            return transform.transOut(yP, self.statY)
        else:
            return None

    def transOutYC(self, ycP):
        if ycP.shape[-1] != 0:
            return transform.transOut(ycP, self.statYC)
        else:
            return None
