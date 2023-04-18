import importlib
from hydroDL.data import gridMET
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse
"""
convert raw data to tab format of each sites
"""

rawFolder = os.path.join(kPath.dirUsgs, 'gridMet', 'raw')
saveFolder = os.path.join(kPath.dirUsgs, 'gridMet', 'output')

# create mask for all USGS basins: gridMetMask.py
# extract gridMet data for all USGS basins: gridMetExtract.py

# setup information
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = range(1979, 2023)

nSite = 9067
iSLst = list(range(0, nSite, 1500))
iELst = iSLst[1:]+[nSite]

for iS, iE in zip(iSLst, iELst):
    errLst = list()
    dataDict = dict()
    for var in varLst:
        tempLst = list()
        for yr in yrLst:
            print('reading {} year {} {}-{}'.format(var, yr, iS, iE))
            fileName = os.path.join(
                rawFolder, '{}_{}_{}_{}.csv'.format(var, yr, iS, iE))
            temp = pd.read_csv(fileName, index_col=0)
            tempLst.append(temp)
        pdf = pd.concat(tempLst)
        dataDict[var] = pdf

    siteNoLst = pdf.columns.tolist()
    t = pd.to_datetime(pdf.index).values.astype('datetime64[D]')

    for k,siteNo in enumerate(siteNoLst):
        temp = dict()
        temp['date'] = t
        for var in varLst:
            temp[var] = dataDict[var][siteNo]
        pdfSite = pd.DataFrame.from_dict(temp).set_index('date')
        pdfSite.to_csv(os.path.join(saveFolder, siteNo))
        print('writing {} {}'.format(k,siteNo))
        if pd.isna(pdfSite).all().all():
            errLst.append(siteNo)

    dfErr = pd.DataFrame(data=errLst)
    dfErr.to_csv(os.path.join(kPath.dirUsgs, 'gridMet',
                              'errLst_{}_{}'.format(iS, iE)), header=False, index=False)


""" slurm script
from hydroDL.master import slurm
codePath = os.path.join(kPath.dirCode, 'app',
                        'data', 'gridMet', 'gridMetFromRaw.py')
cmdLine = 'python -u {}'.format(codePath)
jobName = 'gridMetFromRaw'
slurm.submitJob(jobName, cmdLine, nH=8, nM=32)
"""
