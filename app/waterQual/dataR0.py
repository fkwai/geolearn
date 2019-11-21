import os
import matplotlib.pyplot as plt
import numpy as np
from hydroDL.data import usgs
from hydroDL.post import plot

nwisCode = ['00060', '00608', '00625', '00631', '00665', '80154']
workDir = r'C:\Users\geofk\work\waterQuality'
# site with relative complete records
siteLst = ['04086120', '04186500', '0418810']

# download Daily
for siteNo in siteLst:
    saveFile = os.path.join(workDir, 'data', 'dailyTS', siteNo)
    # usgs.downloadDaily(siteNo, nwisCode, saveFile)

# download Samples
for siteNo in siteLst:
    saveFile = os.path.join(workDir, 'data', 'sample', siteNo)
    # usgs.downloadSample(siteNo, saveFile)

# read data
siteNo = siteLst[0]
dataFolder = r'C:\Users\geofk\work\waterQuality\tsDaily\r0\data'
dfDaily = usgs.readUsgsText(os.path.join(workDir, 'data', 'dailyTS', siteNo),
                            dataType='dailyTS')
dfSample = usgs.readUsgsText(os.path.join(workDir, 'data', 'sample', siteNo),
                             dataType='sample')

# plot time series
fig, axes = plt.subplots(len(nwisCode), 1)
for i, code in enumerate(nwisCode):
    t1 = dfDaily['datetime'].values
    y1 = dfDaily[code].values
    t2 = dfSample['datetime'].values
    y2 = dfSample[code].values
    plot.plotTS([t1, t2], [y1, y2],
                ax=axes[i],
                cLst='rb',
                mLst=[None, '*'],
                lsLst=['-', ':'])
fig.show()