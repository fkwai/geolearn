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

lstWaterQuality = ['DRAIN_SQKM', 'SNOW_PCT_PRECIP', 'GEOL_REEDBUSH_DOM',
                   'STREAMS_KM_SQ_KM', 'PCT_1ST_ORDER', 'BFI_AVE', 'CONTACT',
                   'FORESTNLCD06', 'PLANTNLCD06', 'NUTR_BAS_DOM',
                   'ECO3_BAS_DOM', 'HLR_BAS_DOM_100M', 'ELEV_MEAN_M_BASIN',
                   'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE', 'SLOPE_PCT']
varLst = lstWaterQuality
varLstEx = [
    'DRAIN_SQKM',
    'LAT_GAGE',
    'LNG_GAGE',
    'HYDRO_DISTURB_INDX',
    'BAS_COMPACTNESS',
    'FST32F_BASIN',
    'LST32F_BASIN',
    'WD_SITE',
    'SNOW_PCT_PRECIP',
    'PRECIP_SEAS_IND',
    'GEOL_REEDBUSH_DOM',
    'GEOL_HUNT_DOM_CODE',
    'STREAMS_KM_SQ_KM',
    'STRAHLER_MAX',
    'ARTIFPATH_PCT',
    'HIRES_LENTIC_PCT',
    'BFI_AVE',
    'TOPWET',
    'CONTACT',
    'RUNAVE7100',
    'DDENS_2009',
    'STOR_NID_2009',
    'STOR_NOR_2009',
    'RAW_DIS_NEAREST_DAM',
    'PCT_IRRIG_AG',
    'PCT_1ST_ORDER',
    'FRESHW_WITHDRAWAL',
    'FORESTNLCD06',
    'PLANTNLCD06',
    'DEVNLCD06',
    'MAINS800_DEV',
    'MAINS800_FOREST',
    'MAINS800_PLANT',
    'RIP100_DEV',
    'RIP100_FOREST',
    'RIP100_PLANT',
    'NITR_APP_KG_SQKM',
    'PHOS_APP_KG_SQKM',
    'PESTAPP_KG_SQKM',
    'PDEN_2000_BLOCK',
    'ROADS_KM_SQ_KM',
    'ECO2_BAS_DOM',
    'ECO3_BAS_DOM',
    'NUTR_BAS_DOM',
    'HLR_BAS_DOM_100M',
    'PERMAVE',
    'WTDEPAVE',
    'ROCKDEPAVE',
    'BDAVE',
    'OMAVE',
    'KFACT_UP',
    'RFACT',
    'SLOPE_PCT',
    'ELEV_MEAN_M_BASIN',
    'RRMEAN'
]
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
        # potential issue - new string keys that are not coded
    with open(fileCode, 'w') as fp:
        json.dump(dictCode, fp, indent=4)
    return pdf.replace(dictCode)


def removeField(pdf):
    # remove some fields that are not necessary
    rmColLst = ['REACHCODE', 'STANAME']
    for yr in range(1950, 2010):
        rmColLst.append('PPT{}_AVG'.format(yr))
        rmColLst.append('TMP{}_AVG'.format(yr))
    for yr in range(1900, 2010):
        rmColLst.append('wy{}'.format(yr))
    monthLst = ['JAN', 'FEB', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for m in monthLst:
        rmColLst.append('{}_PPT7100_CM'.format(m))
        rmColLst.append('{}_TMP7100_DEGC'.format(m))
    dfG = pdf.drop(rmColLst, axis=1)
    return dfG


def updateRegion(dfG):
    # PNV2 - merge to upper classes
    fileT = os.path.join(dirTab, 'lookupPNV.csv')
    tabT = pd.read_csv(fileT).set_index('PNV_CODE')
    for code in range(1, 63):
        siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
        dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']
    # Eco3
    fileT = os.path.join(dirTab, 'lookupEco.csv')
    tabT = pd.read_csv(fileT).set_index('Eco3code')
    for code in range(1, 85):
        siteNoTemp = dfG[dfG['ECO3_BAS_DOM'] == code].index
        eco3 = tabT.loc[code]['Eco3']
        EcoB1, EcoB2, EcoB3 = eco3.split('.')
        dfG.at[siteNoTemp, 'EcoB1'] = int(EcoB1)
        dfG.at[siteNoTemp, 'EcoB2'] = int(EcoB2)
        dfG.at[siteNoTemp, 'EcoB3'] = int(EcoB3)
    return dfG


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
