from hydroDL.data import Dataframe, DataModel
from hydroDL import utils
import time
from hydroDL import kPath
import os
import pickle
from hydroDL.data import Dataframe, DataModel
from hydroDL.data import transform, gageII, usgs, gridMET, ntn, GLASS
from hydroDL import kPath
import time
import os
import time
import pandas as pd
import numpy as np
import json


def caseFolder(caseName):
    return os.path.join(kPath.dirVeg, 'model', 'data', caseName)


class DataFrameVeg:
    def __init__(self, caseName):
        saveFolder = caseFolder(caseName)
        self.saveFolder = saveFolder
        npz = np.load(os.path.join(saveFolder, 'data.npz'))
        self.varX = npz['varX'].tolist()
        self.varY = [npz['varY'].tolist()] # adhoc
        self.varXC = npz['varXC'].tolist()
        self.x = npz['x']
        self.y = npz['y']
        self.xc = npz['xc']
        info = pd.read_csv(os.path.join(saveFolder, 'info.csv'))
        self.siteIdLst = info['siteId'].tolist()
        self.t = npz['t']
        self.lat = info['lat'].values
        self.lon = info['lon'].values

    def loadSubset(self, name):
        subsetFile = os.path.join(self.saveFolder, 'subset.json')
        with open(subsetFile, 'r') as f:
            subset = json.load(f)
        indS = subset[name]
        return indS


class DataModelVeg(DataModel):
    def __init__(self, dataFrame, **kw):
        super(DataModelVeg, self).__init__()
        if type(dataFrame) is str:
            DF = DataFrameVeg(dataFrame)
        elif utils.sameClass(type(dataFrame), DataFrameVeg):
            DF = dataFrame
        else:
            raise Exception('unknown DataFrame')
        dictD = dict(
            subset='all',
            varX=DF.varX,
            varXC=DF.varXC,
            varY=DF.varY,
            varYC=None,
        )
        dictD.update(kw)
        self.subset = dictD['subset']
        self.strLst = ['X', 'XC', 'Y', 'YC']
        self.varX = dictD['varX']
        self.varY = dictD['varY']
        self.varXC = dictD['varXC']
        self.varYC = dictD['varYC']
        indS = DF.loadSubset(self.subset)
        self.X = DF.x[:, indS, :]
        self.Y = DF.y[:, indS, None]
        self.XC = DF.xc[indS, :]
        self.YC = None

    def trans(self, *, mtdX=None, mtdXC=None, mtdY=None, mtdYC=None):
        mtdLst = [mtdX, mtdXC, mtdY, mtdYC]
        varLst = [self.varX, self.varXC, self.varY, self.varYC]
        statLst = list()
        for mtd, var in zip(mtdLst, varLst):
            if mtd is None:
                # mtd = io.extractVarMtd(var)
                mtd = None if var is None else ['skip' for x in var]
            statLst.append(dict(mtdLst=mtd))
        self.transIn(
            statX=statLst[0], statXC=statLst[1], statY=statLst[2], statYC=statLst[3]
        )

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
        self.transIn(statX=DM.statX, statXC=DM.statXC, statY=DM.statY, statYC=DM.statYC)

    def getDataRaw(self):
        return self.X, self.XC, self.Y, self.YC

    def save(self):
        pass

    def load(self):
        pass

    def saveStat(self, saveFolder):
        for s in self.strLst:
            stat = getattr(self, 'stat' + s)
            statFile = os.path.join(saveFolder, 'stat' + s)
            with open(statFile, 'wb') as f:
                pickle.dump(stat, f)

    def loadStat(self, saveFolder):
        for s in self.strLst:
            dataRaw = getattr(self, s)
            if dataRaw is not None:
                statFile = os.path.join(saveFolder, 'stat' + s)
                with open(statFile, 'rb') as f:
                    stat = pickle.load(f)
                    setattr(self, 'stat' + s, stat)
        self.transIn(
            statX=self.statX, statXC=self.statXC, statY=self.statY, statYC=self.statYC
        )

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
