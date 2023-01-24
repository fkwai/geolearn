import os
from hydroDL import kPath
from hydroDL.master import slurm

codePath = os.path.join(kPath.dirCode, 'app',
                        'data', 'usgsSample', 'downloadCQ.py')
cmdPtn = 'python {} -iS {} -iE {}'
nSite = 9067
iSLst = list(range(0, nSite, 1500))
iELst = iSLst[1:]+[nSite]
for iS, iE in zip(iSLst, iELst):
    cmdLine = cmdPtn.format(codePath, iS, iE)
    jobName = 'download_{}_{}'.format(iS, iE)
    slurm.submitJob(jobName, cmdLine, nH=8, nM=16)
