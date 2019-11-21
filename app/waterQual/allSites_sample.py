# read inventory of all sites
from hydroDL.data import usgs

# read site inventory
fileName = r'C:\Users\geofk\work\waterQuality\inventory_NWIS_sample.txt'
siteAll = usgs.readUsgsText(fileName)