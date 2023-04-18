from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json

# siteNo = '02171700'
# saveFile = 'temp'
# dfSite = gageII.readData(varLst=['STATE'])
# state = dfSite.loc[siteNo]['STATE']
# # usgs.downloadDaily(siteNo, ['00060'], state, saveFile)
# usgs.downloadDaily(siteNo, ['00060'], saveFile)

# siteNo = '02489500'
# saveFile = 'temp'
# dfSite = gageII.readData(varLst=['STATE'])
# state = dfSite.loc[siteNo]['STATE']
# usgs.downloadSample(siteNo, saveFile)

import hydroDL.data.usgs.read as read
import os
import hydroDL.kPath as kPath
import pandas as pd

siteNo='02294655'
# siteNo = '04161820'
# dfC, dfCF = read.readSampleRaw(siteNo)


fileC = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
dfC = read.readUsgsText(fileC, dataType='sample')

with open(fileC) as f:
    k = 0
    line = f.readline()
    while line[0] == '#':
        line = f.readline()
        k = k + 1
    headLst = line[:-1].split('\t')
    typeLst = f.readline()[:-1].split('\t')
pdf = pd.read_table(fileC, header=k, index_col=None, dtype=str)
pdf.drop(0)
