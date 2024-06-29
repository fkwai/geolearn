import ee


def sentinel1(sd, ed, product="COPERNICUS/S1_GRD"):
    col = (
        ee.ImageCollection(product)
        .filterDate(sd, ed)
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    return col


def landsat8(sd, ed, product="LANDSAT/LC08/C02/T1_L2"):
    col = (
        ee.ImageCollection(product).filterDate(sd, ed)
        # .select(('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'))
        # .filter(ee.Filter.eq('CLOUD_COVER', 0))
    )
    return col


def MCD15A3H(sd, ed, product="MODIS/006/MCD15A3H"):
    col = ee.ImageCollection(product).filterDate(sd, ed)
    return col


def MYD09GA(sd, ed, product="MODIS/061/MYD09GA"):
    col = ee.ImageCollection(product).filterDate(sd, ed)
    return col


def MOD09GA(sd, ed, product="MODIS/061/MOD09GA"):
    col = ee.ImageCollection(product).filterDate(sd, ed)
    return col


def getCol(product, sd, ed):
    col = ee.ImageCollection(product).filterDate(sd, ed)
    return col
