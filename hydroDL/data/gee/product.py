import ee


def sentinel1(sd, ed, product='COPERNICUS/S1_GRD'):
    col = (
        ee.ImageCollection(product)
        .filterDate(sd, ed)
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    )
    return col


def landset8(sd, ed, product='LANDSAT/LC08/C01/T1_SR'):
    col = (
        ee.ImageCollection(product).filterDate(sd, ed)
        # .filter(ee.Filter.eq('CLOUD_COVER', 0))
    )
    return col
