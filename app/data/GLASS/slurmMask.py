import os
from hydroDL import kPath
from hydroDL.master import slurm

# create mask for gageII basins
codePath = os.path.join(kPath.dirCode, 'app',
                        'data', 'GLASS', 'createMask.py')
cmdPtn = 'python {} -S {} -E {}'
nSite = 7111
iSLst = list(range(0, nSite, 500))
iELst = iSLst[1:]+[nSite]
for iS, iE in zip(iSLst, iELst):
    cmdLine = cmdPtn.format(codePath, iS, iE)
    jobName = 'mask_{}_{}'.format(iS, iE)
    slurm.submitJob(jobName, cmdLine, nH=12, nM=16)
# os.system(cmdLine)
