import os
import pandas as pd
import numpy as np
from hydroDL.utils import grid
from .io import writeDataConst

def readDBinfo(*, rootDB, subset):
    if type(subset) is list:
        indSubLst = list()
        rootNameLst = list()
        for k in range(len(subset)):
            rootNameTemp, indSubTemp = readSubset(
                rootDB=rootDB, subset=subset[k])
            indSubLst.append(indSubTemp)
            rootNameLst.append(rootNameTemp)
        if len(set(rootNameLst)) == 1:
            indSub = np.concatenate(indSubLst, axis=0)
            rootName = rootNameLst[0]
        else:
            raise Exception('do not support for multiple root of subset')
    else:
        rootName, indSub = readSubset(rootDB=rootDB, subset=subset)

    crdFile = os.path.join(rootDB, rootName, "crd.csv")
    crdRoot = pd.read_csv(crdFile, dtype=np.float, header=None).values

    indAll = np.arange(0, crdRoot.shape[0], dtype=np.int64)
    if np.array_equal(indSub, np.array([-2])):
        indSub = indAll
        indSkip = None
    else:
        indSub = indSub
        indSkip = np.delete(indAll, indSub)
    crd = crdRoot[indSub, :]
    return rootName, crd, indSub, indSkip


def readSubset(*, rootDB, subset):
    subsetFile = os.path.join(rootDB, "Subset", subset + ".csv")
    print('reading subset ' + subsetFile)
    dfSubset = pd.read_csv(subsetFile, dtype=np.int64, header=0)
    rootName = dfSubset.columns.values[0]
    indSub = dfSubset.values.flatten() - 1
    return rootName, indSub


def divideSubset(df, subset, var=None, varC=None):
    rootName, crd, indSub, indSkip = readDBinfo(
        rootDB=df.rootDB, subset=subset)
    if rootName != df.subset:
        lat1 = crd[:, 0]
        lon1 = crd[:, 1]
        lat2, lon2 = df.getGeo()
        ind1, indSub = grid.intersectGrid(lat1, lon1, lat2, lon2)
        if len(ind1) != len(lat1):
            raise Exception('Root dataset does not cover subset')
    if varC is not None:
        if type(varC) is not list:
            varC = [varC]
        data = df.getDataConst(varC, doNorm=False, rmNan=False)
        dataSub = data[indSub, :]
        len(varC)
        for k in range(len(varC)):
            writeDataConst(dataSub[:, k], varC[k],
                           rootDB=df.rootDB, subset=subset)
