from hydroDL.data import Dataframe, DataModel
from hydroDL import utils
from .io import *
from .subset import *
from . import SMAP


class DataframeCsv(Dataframe):
    def __init__(self, rootDB, *, subset, tRange):
        super(DataframeCsv, self).__init__()
        self.rootDB = rootDB
        self.subset = subset
        rootName, crd, indSub, indSkip = readDBinfo(
            rootDB=rootDB, subset=subset)
        self.lat = crd[:, 0]
        self.lon = crd[:, 1]
        self.indSub = indSub
        self.indSkip = indSkip
        self.rootName = rootName
        self.time = utils.time.tRange2Array(tRange)

    def getDataTs(self, varLst, *, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        yrLst, tDb = SMAP.t2yrLst(self.time)
        indDb, ind = utils.time.intersect(tDb, self.time)
        nt = len(tDb)
        ngrid = len(self.indSub)
        nvar = len(varLst)
        data = np.ndarray([ngrid, nt, nvar])

        # time series
        for k in range(nvar):
            dataTemp = readDataTS(
                rootDB=self.rootDB,
                rootName=self.rootName,
                indSub=self.indSub,
                indSkip=self.indSkip,
                yrLst=yrLst,
                fieldName=varLst[k])
            if doNorm is True:
                stat = readStat(
                    rootDB=self.rootDB, fieldName=varLst[k], isConst=False)
                dataTemp = transNorm(dataTemp, stat)
            data[:, :, k] = dataTemp
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        dataOut = data[:, indDb, :]
        return dataOut

    def getDataConst(self, varLst, *, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        ngrid = len(self.indSub)
        nvar = len(varLst)
        data = np.ndarray([ngrid, nvar])
        for k in range(nvar):
            dataTemp = readDataConst(
                rootDB=self.rootDB,
                rootName=self.rootName,
                indSub=self.indSub,
                indSkip=self.indSkip,
                fieldName=varLst[k])
            if doNorm is True:
                stat = readStat(
                    rootDB=self.rootDB, fieldName=varLst[k], isConst=True)
                dataTemp = transNorm(dataTemp, stat)
            data[:, k] = dataTemp
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def saveDataConst(self, data, fieldName, *, ndigit=8, bCalStat=True):
        writeDataConst(data, fieldName, rootDB=self.rootDB,
                       subset=self.subset, ndigit=ndigit, bCalStat=bCalStat)

    def transform(self, data, fieldLst, *,  toNorm=True, opt='data'):
        if type(fieldLst) is not list:
            fieldLst = [fieldLst]
        out = np.ndarray(data.shape)
        for k in range(len(fieldLst)):
            temp = data[:, :, k]
            isConst = True if temp.shape[1] == 1 else False
            stat = readStat(rootDB=self.rootDB,
                            fieldName=fieldLst[k], isConst=isConst)
            if opt == 'data':
                out[:, :, k] = transNorm(temp, stat, toNorm=toNorm)
            elif opt == 'sigma':
                out[:, :, k] = transNormSigma(temp, stat, toNorm=toNorm)
        return out

    def subsetData(self, subset, *, var=None, varC=None):
        divideSubset(self, subset=subset, var=var, varC=varC)

    def subsetInit(self, subset, *, ind=None):
        rootName, indSub = readSubset(rootDB=self.rootDB, subset=subset)
        if indSub is None:
            writeSubset(rootDB=self.rootDB, rootName=self.subset,
                        subset=subset, ind=ind)
        else:
            raise Exception(
                'Subset existed. TODO: check if the existed subset and the wrtting one are identical')


class DataModelCsv(DataModel):
    def __init__(self,
                 *,
                 rootDB, subset, varT, varC, target, tRange, doNorm=[True, True], rmNan=[True, False], daObs=0):
        super(DataModelCsv, self).__init__()
        df = DataframeCsv(rootDB=rootDB, subset=subset, tRange=tRange)

        self.x = df.getDataTs(varLst=varT, doNorm=doNorm[0], rmNan=rmNan[0])
        self.y = df.getDataTs(varLst=target, doNorm=doNorm[1], rmNan=rmNan[1])
        self.c = df.getDataConst(varLst=varC, doNorm=doNorm[0], rmNan=rmNan[0])

    def getData(self):
        return self.x, self.y, self.c


varTarget = ['SMAP_AM']
varForcing = [
    'APCP_FORA', 'DLWRF_FORA', 'DSWRF_FORA', 'TMP_2_FORA', 'SPFH_2_FORA',
    'VGRD_10_FORA', 'UGRD_10_FORA'
]
varSoilM = [
    'APCP_FORA', 'DLWRF_FORA', 'DSWRF_FORA', 'TMP_2_FORA', 'SPFH_2_FORA',
    'VGRD_10_FORA', 'UGRD_10_FORA', 'SOILM_0-10_NOAH'
]
varConst = [
    'Bulk', 'Capa', 'Clay', 'NDVI', 'Sand', 'Silt', 'flag_albedo',
    'flag_extraOrd', 'flag_landcover', 'flag_roughness', 'flag_vegDense',
    'flag_waterbody'
]
varForcingGlobal = ['GPM', 'Wind', 'Tair', 'Psurf', 'Qair', 'SWdown', 'LWdown']
varSoilmGlobal = [
    'SoilMoi0-10', 'GPM', 'Wind', 'Tair', 'Psurf', 'Qair', 'SWdown', 'LWdown'
]
varConstGlobal = [
    'Bulk', 'Capa', 'Clay', 'NDVI', 'Sand', 'Silt', 'flag_albedo',
    'flag_extraOrd', 'flag_landcover', 'flag_roughness', 'flag_vegDense',
    'flag_waterbody'
]
