
import shapefile

ecoShapeFile = '/mnt/sdc/Kuai/Map/ecoregion/ecoRegionClip.shp'
ecoShape = shapefile.Reader(ecoShapeFile)
ecoIdLst = [ecoShape.records()[x][1] for x in range(17)]
