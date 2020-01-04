import os
import matplotlib.pyplot as plt
import numpy as np
from hydroDL.data import usgs
from hydroDL.post import plot

nwisCode = ['00060', '00608', '00625', '00631', '00665', '80154']
workDir = r'C:\Users\geofk\work\waterQuality'
# siteNo = '04086120'
siteNo = '053416972'

# download data
saveFile = os.path.join(workDir, 'data', 'dailyTS', siteNo)
usgs.downloadDaily(siteNo, nwisCode, saveFile)
saveFile = os.path.join(workDir, 'data', 'sample', siteNo)
usgs.downloadSample(siteNo, saveFile)

# read data
dfDaily = usgs.readUsgsText(os.path.join(workDir, 'data', 'dailyTS', siteNo),
                            dataType='dailyTS')
dfSample = usgs.readUsgsText(os.path.join(workDir, 'data', 'sample', siteNo),
                             dataType='sample')
# forcing data
# see extractForcing.py 

# plot time series
dfCode = usgs.readUsgsText(os.path.join(workDir, 'usgs_parameter_code'))
fig, axes = plt.subplots(len(nwisCode)-1, 1)
for i, code in enumerate(nwisCode[1:]):
    t1 = dfDaily['datetime'].values
    y1 = dfDaily[code].values
    t2 = dfSample['datetime'].values
    y2 = dfSample[code].values
    title = code + ' ' + \
        dfCode['parameter_nm'].loc[dfCode['parameter_cd'] ==code].values[0]
    plot.plotTS([t1, t2], [y1, y2],
                ax=axes[i],
                cLst='rb',
                mLst=[None, '*'],
                lsLst=['-', ':'],
                title=title)
    if i + 1 < len(nwisCode): axes[i].set_xticks([])
fig.show()