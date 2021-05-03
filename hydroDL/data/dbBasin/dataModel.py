from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL import kPath
import time
import os
import time
import pandas as pd
import numpy as np
import json
from . import io
from hydroDL import utils

__all__ = ['DataModelFull']


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
        # init subset
        subsetFile = os.path.join(io.caseFolder(caseName), 'subset.json')
        dictSubset = dict(all=siteNoLst)
        with open(subsetFile, 'w') as fp:
            json.dump(dictSubset, fp, indent=4)
        wqData = cls(caseName)
        return wqData

    def dataProvision(self):
        bq = np.less(self.q, 0., where=~np.isnan(self.q))
        if np.any(bq):
            print('Find negative Q, filled zero')
            self.q[bq] = 0

    def saveSubset(self, nameLst, siteNoLst):
        if type(nameLst) is not list:
            nameLst = [nameLst]
            siteNoLst = [siteNoLst]
        dictNew = dict(zip(nameLst, siteNoLst))
        subsetFile = os.path.join(self.saveFolder, 'subset.json')
        dictSubset = self.loadSubset()
        dictSubset.update(dictNew)
        with open(subsetFile, 'w') as fp:
            json.dump(dictSubset, fp, indent=4)
        self.subset = self.loadSubset()

    def loadSubset(self):
        subsetFile = os.path.join(self.saveFolder, 'subset.json')
        if os.path.exists(subsetFile):
            with open(subsetFile, 'r') as fp:
                dictSubset = json.load(fp)
        else:
            dictSubset = dict(all=self.siteNoLst)
            with open(subsetFile, 'w') as fp:
                json.dump(dictSubset, fp, indent=4)
        return dictSubset

    def getSubsetInd(self, subName, sd, ed):
        indS = [self.siteNoLst.index(siteNo)
                for siteNo in self.subset[subName]]
        sd = np.datetime64(sd)
        ed = np.datetime64(ed)
        if sd < self.sd:
            indT1 = 0
        else:
            indT1 = np.where(self.t == sd)[0][0]
        if ed > self.ed:
            indT2 = len(self.t)
        else:
            indT2 = np.where(self.t == ed)[0][0]+1
        return indT1, indT2, indS

    def extractData(self, varTup, subName, sd, ed):
        dataTup = self.extractVar(varTup)
        outTup = self.extractSubset(dataTup, subName, sd, ed)
        return outTup

    def extractSubset(self, dataTup, subName, sd, ed):
        indT1, indT2, indS = self.getSubsetInd(subName, sd, ed)
        outLst = list()
        for data in dataTup:
            if data is None:
                out = None
            elif data.ndim == 3:
                out = data[indT1:indT2, indS, :]
            elif data.ndim == 2:
                out = data[indS, :]
            outLst.append(out)
        return tuple(outLst)

    def extractVar(self, varTup):
        (varX, varXC, varY, varYC) = varTup
        x = self.extractVarT(varX) if varX is not None else None
        xc = self.extractVarC(varXC) if varXC is not None else None
        y = self.extractVarT(varY) if varY is not None else None
        yc = self.extractVarC(varYC) if varYC is not None else None
        return (x, xc, y, yc)

    def extractVarT(self, varLst):
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

    def extractVarC(self, varLst):
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

    def transIn(self, dataTup, varTup, statTup=None):
        # normalize data in
        if statTup is None:
            [outDataLst, outStatLst] = [list(), list()]
            for (data, var) in zip(dataTup, varTup):
                if data is not None:
                    mtd = io.extractVarMtd(var)
                    outData, outStat = transform.transInAll(data, mtd)
                else:
                    (outData, outStat) = (None, None)
                outDataLst.append(outData)
                outStatLst.append(outStat)
            return outDataLst, outStatLst
        else:
            outDataLst = list()
            for (data, var, stat) in zip(dataTup, varTup, statTup):
                if data is not None:
                    mtd = io.extractVarMtd(var)
                    outData = transform.transInAll(data, mtd, statLst=stat)
                else:
                    outData = None
                outDataLst.append(outData)
            return outDataLst

    def transOut(self, data, stat, var):
        mtd = io.extractVarMtd(var)
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
        t = self.t
        tn = utils.time.date2num(t)
        ns = len(self.siteNoLst)
        sinT = np.sin(2*np.pi*tn/365.24)
        cosT = np.sin(2*np.pi*tn/365.24)
        matT = np.stack([tn, sinT, cosT], axis=1)
        matTE = np.repeat(matT[:, None, :], ns, axis=1)
        self.f = np.concatenate([self.f, matTE], axis=2)
        self.varF.append('datenum')
        self.varF.append('sinT')
        self.varF.append('cosT')
