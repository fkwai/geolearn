from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot


# test the difference between daily value and instant values
# and affect to CQ shape

import hydroDL.data.usgs.read as read

# siteNo = '06670500'
siteNo = '06670500'
dfC, dfCF = read.readSampleRaw(siteNo)
dfC, dfCF = read.readSampleRaw(siteNo)


fileC = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
fileQ = os.path.join(kPath.dirRaw, 'USGS', 'streamflow', siteNo)
dfC = read.readUsgsText(fileC, dataType='sample')
dfQ = read.readStreamflow(siteNo=siteNo)


dfC[['sample_tm']].to_csv('temp')

dfC[['00915', '00060', 'sample_tm', 'sample_dt']].to_csv('temp')

tempC = dfC[['00915', '00060', 'sample_tm','sample_dt']]


tempC=tempC[~tempC['sample_tm'].isna()]
# tempC = tempC[~tempC[['00915', '00060']].isna().all(axis=1)]
tempC=tempC[~tempC['00060'].isna()]


tempC.to_csv('temp')
tempQ = dfQ[['00060_00003','datetime']]
tempC=tempC.merge(tempQ,left_on=['sample_dt'],right_on=['datetime'])

fig,ax=plt.subplots(1,1)
ax.plot(tempQ['datetime'],tempQ['00060_00003'])
ax.plot(tempC['sample_dt'],tempC['00060_00003'],'ko')
ax.plot(tempC['sample_dt'],tempC['00060'],'r*')
fig.show()
# dataName='dbAll'
# DF = dbBasin.DataFrameBasin(dataName)
