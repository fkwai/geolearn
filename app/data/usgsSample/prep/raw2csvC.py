
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
# concentrations
dirC = os.path.join(kPath.dirUsgs, 'sample', 'csvAll')
if not os.path.exists(dirC):
    os.mkdir(dirC)
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfC, dfCF = read.readSampleRaw(siteNo)
    if dfC is not None:
        dfC.to_csv(os.path.join(dirC, siteNo))
        dfCF.to_csv(os.path.join(dirC, siteNo+'_flag'))
    print('{}/{} {} {:.2f}'.format(i, siteNo, len(siteNoLst), time.time()-t0))

""" slurm script
from hydroDL.master import slurm
codePath = os.path.join(kPath.dirCode, 'app',
                        'data', 'usgsSample', 'raw2csvC.py')
cmdLine = 'python {}'.format(codePath)
jobName = 'raw2csvC'
slurm.submitJob(jobName, cmdLine, nH=8, nM=16)
"""

# # read all C/Q data and save as csv - improve future efficiency
# # streamflow
# dirQ = os.path.join(kPath.dirData, 'USGS', 'streamflow', 'csv')
# t0 = time.time()
# for i, siteNo in enumerate(siteNoLst):
#     dfQ = usgs.readStreamflow(siteNo)
#     if dfQ is not None:
#         dfQ.to_csv(os.path.join(dirQ, siteNo))
#     print('\t {}/{} {:.2f}'.format(
#         i, len(siteNoLst), time.time()-t0), end='\r')

