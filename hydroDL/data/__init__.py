"""
:dataset: db.dataset is a container of data
"""

extentCONUS = [-125, -65, 25, 50]

from hydroDL.data import transform
import os
import pickle


class Dataframe(object):
    def getGeo(self, ndigit=8):
        # return self.lat, self.lon
        pass

    def getT(self):
        # return self.time
        pass


class DataModel(object):
    def __init__(
        self,
        X=None,
        XC=None,
        Y=None,
        YC=None,
        statX=None,
        statXC=None,
        statY=None,
        statYC=None,
        mtdX=None,
        mtdXC=None,
        mtdY=None,
        mtdYC=None,
    ):
        self.X, self.XC, self.Y, self.YC = (X, XC, Y, YC)
        self.statX, self.statXC, self.statY, self.statYC = (
            statX,
            statXC,
            statY,
            statYC,
        )
        self.mtdX, self.mtdXC, self.mtdY, self.mtdYC = (mtdX, mtdXC, mtdY, mtdYC)
        self.nx = 0 if X is None else X.shape[-1]
        self.nxc = 0 if XC is None else XC.shape[-1]
        self.ny = 0 if Y is None else Y.shape[-1]
        self.nyc = 0 if YC is None else YC.shape[-1]
        self.x, self.xc, self.y, self.yc = None, None, None, None

    def trans(self, mtdDefault='skip'):
        mtdLst = [self.mtdX, self.mtdXC, self.mtdY, self.mtdYC]
        nLst = [self.nx, self.nxc, self.ny, self.nyc]
        statLst = list()
        for mtd, n in zip(mtdLst, nLst):
            if mtd is None:
                mtd = None if n == 0 else [mtdDefault for x in range(n)]
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

    def borrowStat(self, DM):
        self.transIn(statX=DM.statX, statXC=DM.statXC, statY=DM.statY, statYC=DM.statYC)

    def getData(self):
        return self.x, self.xc, self.y, self.yc

    def getDataRaw(self):
        return self.X, self.XC, self.Y, self.YC

    def getVar(self):
        return self.varX, self.varXC, self.varY, self.varYC

    def getStat(self):
        return self.statX, self.statXC, self.statY, self.statYC

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
