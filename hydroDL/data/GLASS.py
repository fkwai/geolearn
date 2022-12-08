import os
import numpy as np
import pandas as pd
from hydroDL import kPath

varLst = ['LAI', 'FAPAR', 'NPP']

dictStat = dict(LAI='norm', FAPAR='norm', NPP='norm')


def readBasin(siteNo, varLst=varLst, freq='D'):
    """read basin averaged forcing data, plenty of work is done before. See:
    app\data\GLASS\creatMask.py
    app\data\GLASS\extractData.py
    app\data\GLASS\tempData.py
    app\data\GLASS\tempMask.py    
    Arguments:
        siteNo {str} -- usgs site number
    Returns:
        pandas.Dataframe -- output table
    """
    if freq == 'R':  # raw frequency
        dirF = os.path.join(kPath.dirUSGS, 'GLASS', 'output')
    if freq == 'D':
        dirF = os.path.join(kPath.dirUSGS, 'GLASS', 'Daily')
    if freq == 'W':
        dirF = os.path.join(kPath.dirUSGS, 'GLASS', 'Weekly')
    fileF = os.path.join(dirF, siteNo)
    dfF = pd.read_csv(fileF)
    dfF['date'] = pd.to_datetime(dfF['date'], format='%Y-%m-%d')
    dfF = dfF.set_index('date')
    return dfF[varLst]
