import os
import numpy as np
import pandas as pd
from hydroDL import kPath

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

dictStat = dict(pr='log-norm', sph='norm', srad='norm',
                tmmn='norm', tmmx='norm', pet='norm', etr='norm')


def readBasin(siteNo, varLst=varLst):
    """read basin averaged forcing data, plenty of work is done before. See:
        app\waterQual\data\gridMetExtract.py
        app\waterQual\data\gridMetFromRaw.py
        app\waterQual\data\gridMetMask.py
    Arguments:
        siteNo {str} -- usgs site number
    Returns:
        pandas.Dataframe -- output table
    """
    fileF = os.path.join(kPath.dirUsgs, 'gridMet', 'output', siteNo)
    dfF = pd.read_csv(fileF)
    dfF['date'] = pd.to_datetime(dfF['date'], format='%Y-%m-%d')
    dfF = dfF.set_index('date')
    return dfF[varLst]
