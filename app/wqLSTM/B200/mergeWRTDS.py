
import os
from hydroDL import kPath
from hydroDL.data import dbBasin
import pandas as pd
import numpy as np

dataName = 'NY5'
trainLst = ['B15']+['rmYr5b{}'.format(k) for k in range(5)]+['rmRT5b{}'.format(k) for k in range(5)]

DF = dbBasin.DataFrameBasin(dataName)

dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')

for trainSet in trainLst:
    folderName = '{}-{}-{}'.format(dataName, trainSet, 'all')
    yWLst=list()
    for k,siteNo in enumerate(DF.siteNoLst):
        print('reading {} {}'.format(k,siteNo))
        fileName=os.path.join(dirWRTDS,folderName,siteNo)
        dfW=pd.read_csv(fileName,index_col=0)
        yWLst.append(dfW.values)

    yW=np.stack(yWLst,axis=-1).swapaxes(1,2)
    np.savez_compressed(os.path.join(dirWRTDS, folderName), yW=yW)
