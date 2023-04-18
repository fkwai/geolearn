
from hydroDL import kPath
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os
import hydroDL.data.usgs.read as read
import json

dirUSGS = kPath.dirUsgs
siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
siteNoLst=dfSite['siteNo'].tolist()

# read all C/Q data and save as csv - improve future efficiency
# concentrations
dirC = os.path.join(kPath.dirUsgs, 'sample', 'csvAll')
if not os.path.exists(dirC):
    os.mkdir(dirC)

dictErrC = dict()
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    try:
        fileCsv=os.path.join(dirC, siteNo)
        if not os.path.exists(fileCsv):
            dfC, dfCF = read.readSampleRaw(siteNo)
            if dfC is not None:
                dfC.to_csv(fileCsv)
                dfCF.to_csv(fileCsv+'_flag')
            print('{}/{} {} {:.2f}'.format(i, len(siteNoLst),siteNo, time.time()-t0))
    except Exception as e:
        dictErrC[siteNo]=str(e)
        print('error site Q {}'.format(siteNo))
errFileC = os.path.join(kPath.dirRaw, 'USGS', 'errC-raw2csv.csv')
with open(errFileC+'.json', 'w') as f:
    json.dump(dictErrC, f)
