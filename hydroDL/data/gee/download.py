import os
from . import geeutils
import time


def download(imgCol, outFolder, scale, lat, lon, nameLst):
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
    # for ind, row in tabCrd.iterrows():
    for k, siteDict in enumerate(dictLst):
        t0 = time.time()
        lat = siteDict['crd'][0]
        lon = siteDict['crd'][1]
        siteId = siteDict['siteId']
        fileName = os.path.join(outFolder, siteId + '.csv')
        geometry = ee.Geometry.Point([lon, lat])
        region = imgCol.getRegion(geometry, int(scale)).getInfo()
        df = geeutils.record2df(region)
        # df.drop(columns=['longitude','latitude'], inplace=True)
        df.to_csv(fileName, index=False)
        print(siteId, time.time() - t0)
