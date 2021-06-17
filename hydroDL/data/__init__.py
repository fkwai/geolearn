"""
:dataset: db.dataset is a container of data
"""


class Dataframe(object):
    
    def getGeo(self, ndigit=8):
        # return self.lat, self.lon
        pass

    def getT(self):
        # return self.time
        pass


class DataModel():
    def __init__(self):
        self.x, self.xc, self.y, self.yc = (None, None, None, None)
        self.statX, self.statXC, self.statY, self.statYC = (
            None, None, None, None)
        self.varX, self.varXC, self.varY, self.varYC = (None, None, None, None)

    def getDataTrain(self):
        # return self.x, self.y, self.c
        pass

    def getData(self):
        return self.x, self.xc, self.y, self.yc

    def getVar(self):
        return self.varX, self.varXC, self.varY, self.varYC

    def getStat(self):
        return self.statX, self.statXC, self.statY, self.statYC
