import os
import numpy as np
import pandas as pd
from hydroDL import kPath

varLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K',
          'Na', 'NH4', 'NO3', 'Cl', 'SO4']
flagLst = ['flagCa', 'flagMg', 'flagK', 'flagNa', 'flagNH4',
           'flagNO3', 'flagCl', 'flagSO4', 'valcode', 'invalcode']

dictStat = dict(ph='norm', Conduc='norm', Ca='log-norm', Mg='log-norm',
                K='log-norm', Na='log-norm', NH4='log-norm', NO3='log-norm',
                Cl='log-norm', SO4='log-norm', distNTN='norm')

dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')


def readDataRaw():
    """read downloaded NTN raw data:
    'C:\\Users\\geofk\\work\\database\\EPA\\NTN\\NTN-All-w.csv'
    'C:\\Users\\geofk\\work\\database\\EPA\\NTN\\NTNsites.csv'
    """
    # read raw data
    fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
    tabData = pd.read_csv(fileData)
    # fix the data
    tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
    tabData = tabData.replace(-9, np.nan)
    return tabData


def loadSite():
    fileSite = os.path.join(dirNTN, 'NTNsites.csv')
    tabSite = pd.read_csv(fileSite)
    tabSite = tabSite.rename(columns={'siteid': 'siteID'})
    tabSite.set_index('siteID', inplace=True)
    crdNTN = pd.read_csv(os.path.join(
        dirNTN, 'crdNTN.csv'), index_col='siteid')
    crdNTN.index.rename('siteID', inplace=True)
    tabSite = tabSite.join(crdNTN[['x', 'y']])
    # find valid siteNo
    # siteIdLst1 = tabData['siteID'].unique().tolist()
    # siteIdLst2 = tabSite['siteID'].tolist()
    # siteIdLst = list(set(siteIdLst1).intersection(siteIdLst2))
    # tabSite = tabSite[tabSite['siteID'].isin(siteIdLst)].reset_index(drop=True)
    # tabData = tabData[tabData['siteID'].isin(siteIdLst)].reset_index(drop=True)
    tabSite.drop(['CO83', 'NC30', 'WI19'], inplace=True)
    return tabSite


def loadCrdUSGS():
    # crds are projected to USGS proj in arcgis
    dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
    crdUSGS = pd.read_csv(os.path.join(
        dirNTN, 'crdUSGS.csv'), dtype={'STAID': str})
    crdUSGS = crdUSGS.set_index('STAID')
    crdUSGS.index.rename('idUSGS', inplace=True)
    return crdUSGS


def readSite(ntnId, freq='W'):
    if freq == 'D':
        ntnFolder = os.path.join(dirNTN, 'csv', 'daily')
    elif freq == 'W':
        ntnFolder = os.path.join(dirNTN, 'csv', 'weekly')
    fileName = os.path.join(ntnFolder, ntnId)
    tab = pd.read_csv(fileName, index_col='date')
    tab.index = pd.to_datetime(tab.index)
    return tab


def readBasin(siteNo, varLst=varLst, freq='W'):
    dirUsgs = os.path.join(dirNTN, 'usgs')
    if freq == 'W':
        dirNtn = os.path.join(dirUsgs, 'weekly')
    elif freq == 'D':
        dirNtn = os.path.join(dirUsgs, 'daily')
    dfP = pd.read_csv(os.path.join(dirNtn, siteNo), index_col='date')
    dfP.index = pd.to_datetime(dfP.index)
    dfP = dfP[varLst]
    return dfP
