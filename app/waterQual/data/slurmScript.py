import os
from hydroDL import kPath
from hydroDL.master import slurm

# create mask for gageII basins
cmdPtn = os.path.join(kPath.dirCode, 'app', 'waterQual',
                      'data', 'gridMetMask.py') + ' -S {} -E {}'
nSite = 7111
iSLst = list(range(0, nSite, 300))
iELst = iSLst[1:]+[nSite]

for iS, iE in zip(iSLst, iELst):
    cmdLine = cmdPtn.format(iS, iE)
    jobName = 'mask{}_{}'.format(iS, iE)
    slurm.submitJob(jobName, cmdLine, nH=12, nM=16)
