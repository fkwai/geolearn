import pandas as pd
import os
import json
import fiona
from hydroDL import kPath

dirGageII = os.path.join(kPath.dirData, 'gageII')

dirTab = os.path.join(
    dirGageII, 'basinchar_and_report_sept_2011', 'spreadsheets-in-csv-format')
dirShape = os.path.join(dirGageII, 'boundaries-shapefiles-by-aggeco')

lstWaterQuality = ['DRAIN_SQKM', 'SNOW_PCT_PRECIP', 'GEOL_REEDBUSH_DOM', 'STREAMS_KM_SQ_KM', 'PCT_1ST_ORDER',
                   'BFI_AVE', 'CONTACT', 'FORESTNLCD06', 'PLANTNLCD06', 'NUTR_BAS_DOM', 'ECO3_BAS_DOM', 'HLR_BAS_DOM_100M',
                   'ELEV_MEAN_M_BASIN', 'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE', 'SLOPE_PCT']


def readTab(varType):
    if varType == 'Flow_Record':
        varType = 'flowrec'
    fileInv = os.path.join(dirTab, 'conterm_{}.txt'.format(varType))
    tab = pd.read_csv(fileInv, encoding='ISO-8859-1', dtype={'STAID': str})
    return tab


def getVariableDict(varLst=None):
    """ get a dict of ggII variables
    Keyword Arguments:
        varLst {list} -- list of variable names (default: {None})
    Returns:
        dict -- variable type -> list of variable name
    """

    fileDesc = os.path.join(dirTab, 'variable_descriptions.txt')
    tab = pd.read_csv(fileDesc)
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
        if vt == 'Climate_Ppt_Annual':
            vnLst = ['PPT{}_AVG'.format(x) for x in range(1950, 2010)]
        elif vt == 'Climate_Tmp_Annual':
            vnLst = ['TMP{}_AVG'.format(x) for x in range(1950, 2010)]
        else:
            vnLst = tab[tab['VARIABLE_TYPE'] ==
                        vt]['VARIABLE_NAME'].values.tolist()
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
        tab = readTab(key)
        if siteNoLst is None:
            tempLst.append(tab.set_index('STAID').loc[:, value])
        else:
            tempLst.append(tab.set_index('STAID').loc[siteNoLst][value])
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
            strLst.sort()
            codeLst = list(range(len(strLst)))
            dictCode[var] = dict(zip(strLst, codeLst))
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
