import ee
import time
import pandas as pd
from . import utils


def col2lst(imageCol, nImage=None):
    if nImage is None:
        nImage = imageCol.size().getInfo()
    imageLst = imageCol.toList(nImage)
    return imageLst, nImage


def getRegion(imageCol, fieldLst, region, nImage=None):
    imageLst, nImage = col2lst(imageCol, nImage)
    df = pd.DataFrame(columns=['datetime'] + fieldLst)
    t0 = time.time()
    for kk in range(nImage):
        image = ee.Image(imageLst.get(kk))
        temp = image.reduceRegion(ee.Reducer.mean(), region).getInfo()
        tstr = image.date().format('yyyy-MM-dd').getInfo()
        temp['datetime'] = tstr
        df = df.append(temp, ignore_index=True)
        tc = time.time() - t0
        print('\t image {}/{} time cost {:.2f}'.format(kk + 1, nImage, tc),
              end='\r')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    return df
