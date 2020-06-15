import pandas as pd
import numpy as np
import os
import json
# import fiona
from hydroDL import kPath

dirGageII = os.path.join(kPath.dirData, 'gageII')

dirTab = os.path.join(
    dirGageII, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
dirShape = os.path.join(dirGageII, 'boundaries-shapefiles-by-aggeco')

lstWaterQuality = ['DRAIN_SQKM', 'SNOW_PCT_PRECIP', 'GEOL_REEDBUSH_DOM', 'STREAMS_KM_SQ_KM', 'PCT_1ST_ORDER',
                   'BFI_AVE', 'CONTACT', 'FORESTNLCD06', 'PLANTNLCD06', 'NUTR_BAS_DOM', 'ECO3_BAS_DOM', 'HLR_BAS_DOM_100M',
                   'ELEV_MEAN_M_BASIN', 'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE', 'SLOPE_PCT']
varLst = lstWaterQuality
dictStat = dict(
    DRAIN_SQKM='log-norm',
    SNOW_PCT_PRECIP='norm',
    GEOL_REEDBUSH_DOM='norm',
    STREAMS_KM_SQ_KM='norm',
    PCT_1ST_ORDER='norm',
    BFI_AVE='norm',
    CONTACT='log-norm',
    FORESTNLCD06='norm',
    PLANTNLCD06='norm',
    NUTR_BAS_DOM='norm',
    ECO3_BAS_DOM='norm',
    HLR_BAS_DOM_100M='norm',
    ELEV_MEAN_M_BASIN='norm',
    PERMAVE='norm',
    WTDEPAVE='norm',
    ROCKDEPAVE='norm',
    SLOPE_PCT='norm'
)


def readTab(varType):
    if varType == 'Flow_Record':
        varType = 'flowrec'
    fileInv = os.path.join(dirTab, 'conterm_{}.txt'.format(varType.lower()))
    tab = pd.read_csv(fileInv, encoding='ISO-8859-1',
                      dtype={'STAID': str})
    return tab


def getVariableDict(varLst=None):
    """ get a dict of ggII variables - deal with many situations
    Keyword Arguments:
        varLst {list} -- list of variable names (default: {None})
    Returns:
        dict -- variable type -> list of variable name

    code to find strange issues:
    ```
    dictVar = gageII.getVariableDict()
    missLst1 = list()
    missLst2 = list()
    for key, value in dictVar.items():
        tab = gageII.readTab(key).set_index('STAID')
        colLst = tab.columns.tolist()
        v1 = [v for v in value if v not in colLst]
        v2 = [v for v in colLst if v not in value]
        missLst1 = missLst1+v1
        missLst2 = missLst2+v2
    ```
    Flow_Record has a extra comma
    """

    fileDesc = os.path.join(dirTab, 'variable_descriptions.txt')
    tab = pd.read_csv(fileDesc)
    # drop variables in desc
    tab = tab.drop(tab.loc[tab['VARIABLE_NAME'] == 'STAID'].index)
    tab = tab.drop(tab.loc[tab['VARIABLE_TYPE'] == 'X_Region_Names'].index)
    tab = tab.drop(tab.loc[(tab['VARIABLE_TYPE'] == 'BasinID') & (
        tab['VARIABLE_NAME'] == 'DRAIN_SQKM')].index)
    if varLst is not None:
        tab = tab.set_index('VARIABLE_NAME').loc[varLst].reset_index()[
            ['VARIABLE_TYPE', 'VARIABLE_NAME']]
    dictVar = dict()
    vtLst = tab.VARIABLE_TYPE.unique().tolist()
    for vt in vtLst:
        vnLst = tab[tab['VARIABLE_TYPE'] ==
                    vt]['VARIABLE_NAME'].values.tolist()
        # modify var names
        if vt == 'Climate_Ppt_Annual':
            vnLst = ['PPT{}_AVG'.format(x) for x in range(1950, 2010)]
        if vt == 'Climate_Tmp_Annual':
            vnLst = ['TMP{}_AVG'.format(x) for x in range(1950, 2010)]
        if 'FST32F_SITE' in vnLst:
            vnLst[vnLst.index('FST32F_SITE')] = 'FST32SITE'
        if 'LST32F_SITE' in vnLst:
            vnLst[vnLst.index('LST32F_SITE')] = 'LST32SITE'
        if 'wy1900 through wy2009 (110 values)' in vnLst:
            vnLst.remove('wy1900 through wy2009 (110 values)')
            temp = ['wy{}'.format(x) for x in range(1900, 2010)]
            vnLst = vnLst+temp
        dictVar[vt] = vnLst
    return dictVar


def readData(*, varLst=None, siteNoLst=None):
    """ read ggII raw data
    Keyword Arguments:
        varLst {list} -- list of variable names (default: {None})
        siteNoLst {list} -- list of siteNo in str (default: {None})
    Returns:
        pandas.core.frame.DataFrame -- ggII data
    """
    dictVar = getVariableDict(varLst)
    tempLst = list()
    for key, value in dictVar.items():
        tab = readTab(key).set_index('STAID')
        vExist = [v for v in value if v in tab.columns.tolist()]
        temp = tab[vExist]
        if siteNoLst is not None:
            temp = temp.loc[siteNoLst]
        if 'FLOW_PCT_EST_VALUES' in vExist:  # exception
            var = 'FLOW_PCT_EST_VALUES'
            temp[var] = temp[var].replace({'ND': np.nan}).astype(float)
        tempLst.append(temp)
    pdf = pd.concat(tempLst, axis=1)
    return pdf


def updateCode(pdf):
    """update string fields of a ggII dataframe. lookup code will be write into a json file. 
    Arguments:
        pdf {pandas.core.frame.DataFrame} -- input pdf                
    Returns:
        pandas.core.frame.DataFrame -- output pdf
    """
    varTempLst = pdf.select_dtypes(include=['object']).columns.tolist()
    if len(varTempLst) == 0:
        return pdf
    dfTemp = readData(varLst=varTempLst)
    fileCode = os.path.join(dirTab, 'lookupCode.json')
    if os.path.exists(fileCode):
        with open(fileCode, 'r') as fp:
            dictCode = json.load(fp)
    else:
        dictCode = dict()
    for var in varTempLst:
        if var not in dictCode.keys():
            strLst = dfTemp[var].unique().tolist()
            if np.nan in strLst:
                strLst.remove(np.nan)
            strLst.sort()
            codeLst = list(range(len(strLst)))
            dictCode[var] = dict(zip(strLst, codeLst))
            print('added {} - {} unique values'.format(var, len(strLst)))
    with open(fileCode, 'w') as fp:
        json.dump(dictCode, fp, indent=4)
    return pdf.replace(dictCode)


def extractBasins(siteNoLst, outShapeFile):
    """ extract shape of sites    
    Arguments:
        siteNoLst {list} -- list of siteNo in str
        outShapeFile {str} -- output shapefile
    """
    tab = readTab('bas_classif')
    regionLst = list()
    for siteNo in siteNoLst:
        region = tab[tab['STAID'] == siteNo]['AGGECOREGION'].values[0]
        ref = tab[tab['STAID'] == siteNo]['CLASS'].values[0]
        if ref == 'Ref':
            regionLst.append(ref)
        else:
            regionLst.append(region)
    regionSet = set(regionLst)

    shapeDict = dict()
    idDict = dict()
    for region in regionSet:
        if region == 'Ref':
            fileShape = os.path.join(dirShape, 'bas_ref_all.shp')
        else:
            fileShape = os.path.join(dirShape,
                                     'bas_nonref_{}.shp'.format(region))
        shapeDict[region] = fiona.open(fileShape)
        idDict[region] = [
            x['properties']['GAGE_ID'] for x in shapeDict[region]
        ]

    with fiona.open(os.path.join(dirShape, 'bas_ref_all.shp')) as shape:
        meta = shape.meta
    with fiona.open(outShapeFile, 'w', **meta) as output:
        for siteNo, region in zip(siteNoLst, regionLst):
            print('writing {}'.format(siteNo))
            shapeLst = shapeDict[region]
            idLst = idDict[region]
            feat = shapeLst[idLst.index(siteNo)]
            output.write(feat)

    for region in regionSet:
        shapeDict[region].close()
