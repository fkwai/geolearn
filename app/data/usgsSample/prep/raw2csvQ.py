
from hydroDL import kPath
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os
import hydroDL.data.usgs.read as read

dirUSGS = kPath.dirUsgs
siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
siteNoLst=dfSite['siteNo'].tolist()

# read all C/Q data and save as csv - improve future efficiency
dirQ = os.path.join(kPath.dirUsgs,'streamflow', 'csv')
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfQ = read.readStreamflow(siteNo,csv=False)
    if dfQ is not None:
        dfQ.to_csv(os.path.join(dirQ, siteNo))
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')


""" slurm script
from hydroDL.master import slurm
codePath = os.path.join(kPath.dirCode, 'app',
                        'data', 'usgsSample', 'raw2csvQ.py')
cmdLine = 'python {}'.format(codePath)
jobName = 'raw2csvQ'
slurm.submitJob(jobName, cmdLine, nH=8, nM=16)
"""
