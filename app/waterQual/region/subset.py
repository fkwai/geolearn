
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import pandas as pd
import json
import os

regionLst = ['ECO2_BAS_DOM', 'NUTR_BAS_DOM', 'HLR_BAS_DOM_100M', 'PNV_BAS_DOM']
dfG = gageII.readData(varLst=regionLst+['LAT_GAGE', 'LNG_GAGE', 'CLASS'])

# deal with PNV
fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
tabT = pd.read_csv(fileT).set_index('PNV_CODE')
for code in range(1, 63):
    siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
    dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']


dictName = {
    'PNV': 'PNV_BAS_DOM2',
    'NUTR': 'NUTR_BAS_DOM',
    'HLR': 'HLR_BAS_DOM_100M',
    'ECO': 'ECO2_BAS_DOM'}
dictRegion = {
    'PNV': [2, 3, 4, 5, 9, 11],
    'NUTR': [2, 3, 4, 5, 6, 7, 8, 9, 11, 14],
    'HLR': [3, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 20],
    'ECO': [5.3, 6.2, 8.1, 8.2, 8.3, 8.4, 9.2, 9.3, 9.4, 10.1, 11.1]
}


# load data
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
info = wqData.info
info['yr'] = pd.DatetimeIndex(info['date']).year
nameLst = list()
indLst = list()
for region in list(dictRegion.keys()):
    for regionId in dictRegion[region]:
        regionName = dictName[region]
        siteRegionLst = dfG.loc[dfG[regionName] == regionId].index.tolist()
        bs = info['siteNo'].isin(siteRegionLst)
        b1 = (info['yr'] < 2010).values
        b2 = (info['yr'] >= 2010).values
        ind1 = info.index[b1 & bs].values
        ind2 = info.index[b2 & bs].values
        if region == 'ECO':
            idLst = [int(x) for x in str(regionId).split('.')]
            regionId = '{:02d}{:02d}'.format(*idLst)
        else:
            regionId = '{:02d}'.format(regionId)
        nameLst.append('comb-{}{}-B10'.format(region, regionId))
        indLst.append(ind1)
        nameLst.append('comb-{}{}-A10'.format(region, regionId))
        indLst.append(ind2)
        print(region, regionId)
wqData.saveSubset(nameLst, indLst)
