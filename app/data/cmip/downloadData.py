import os
from hydroDL import kPath
from hydroDL.data import cmip
from hydroDL.master import slurm
# write urls to a file
mLst = ['MPI-ESM1-2-XR', 'HadGEM3-GC31-HM', 'FGOALS-f3-H',
        'EC-Earth3P-HR', 'EC-Earth3P', 'CNRM-CM6-1-HR']
for m in mLst:
    for lab in ['hist', 'future']:
        fileName = '{}-{}'.format(m, lab)
        urlFile = os.path.join(
            kPath.dirCode, 'app', 'data', 'cmip', 'urlFile', fileName)
        codePath = os.path.join(kPath.dirCode, 'data', 'cmip', 'download.py')
        cmdLine = 'python {} -F {}'.format(codePath, urlFile)
        slurm.submitJob(fileName, cmdLine, nH=2, nM=16)
# urlFile = os.path.join(
#     kPath.dirCode, 'app', 'data', 'cmip', 'urlFile', 'temp')
# outFolder = os.path.join(kPath.dirRaw, 'CMIP6')
# cmip.download.byUrlFile(urlFile, outFolder)
