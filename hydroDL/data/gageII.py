import pandas as pd
import os
import fiona

dirGageII = r'C:\Users\geofk\work\database\gageII'
dirTab = os.path.join(
    dirGageII, r'basinchar_and_report_sept_2011\spreadsheets-in-csv-format')
dirShape = os.path.join(dirGageII, r'boundaries-shapefiles-by-aggeco')


def readTab(field):
    fileInv = os.path.join(dirTab, 'conterm_{}.txt'.format(field))
    tab = pd.read_csv(fileInv, encoding='ISO-8859-1', dtype={'STAID': str})
    return tab


def extractBasins(siteNoLst, outShapeFile):
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
