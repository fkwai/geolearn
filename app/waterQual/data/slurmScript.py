import os
from hydroDL import kPath
from hydroDL.master import slurm

# create mask for gageII basins
codePath = os.path.join(kPath.dirCode, 'app',
                        'waterQual', 'data', 'gridMetMask.py')
cmdPtn = 'python {} -S {} -E {}'
nSite = 7111
iSLst = list(range(0, nSite, 1000))
iELst = iSLst[1:]+[nSite]
for iS, iE in zip(iSLst, iELst):
    cmdLine = cmdPtn.format(codePath, iS, iE)
    jobName = 'mask{}_{}'.format(iS, iE)
    # slurm.submitJob(jobName, cmdLine, nH=4, nM=16)
# os.system(cmdLine)


# save all mask to one file - too large not worth it
codePath = os.path.join(kPath.dirCode, 'app',
                        'waterQual', 'data', 'gridMetMaskAll.py')
jobName = 'maskAll'
cmdLine = 'python {}'.format(codePath)
# slurm.submitJob(jobName, cmdLine, nH=1, nM=64)

# extract gridMET to raw data (one file for one variable, multiple sites)
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
codePath = os.path.join(kPath.dirCode, 'app',
                        'waterQual', 'data', 'gridMetExtract.py')
cmdPtn = 'python {} -var {} -syr {} -eyr {} -smask {} -emask {}'
nSite = 7111
iSLst = list(range(0, nSite, 2000))
iELst = iSLst[1:]+[nSite]
for var in varLst:
    for iS, iE in zip(iSLst, iELst):
        cmdLine = cmdPtn.format(codePath, var, 1979, 2020, iS, iE)
        jobName = '{}_{}_{}'.format(var, iS, iE)
        # slurm.submitJob(jobName, cmdLine, nH=4, nM=64)

# convert from raw to output
codePath = os.path.join(kPath.dirCode, 'app',
                        'waterQual', 'data', 'gridMetFromRaw.py')
jobName = 'gridMetFromRaw'
cmdLine = 'python {}'.format(codePath)
slurm.submitJob(jobName, cmdLine, nH=2, nM=64)
