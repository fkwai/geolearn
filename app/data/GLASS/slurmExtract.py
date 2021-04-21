import os
from hydroDL import kPath
from hydroDL.master import slurm
import numpy as np

# create mask for gageII basins
codePath = os.path.join(kPath.dirCode, 'app',
                        'data', 'GLASS', 'extractData.py')
cmdPtn = 'python {} -S {} -E {}'
ns = 7111
k1Lst = np.arange(0, ns, 100)
k2Lst = np.append(k1Lst[1:], ns)
nk = len(k1Lst)

iSLst = list(range(0, nk, 6))
iELst = iSLst[1:]+[nk]
for iS, iE in zip(iSLst, iELst):
    cmdLine = cmdPtn.format(codePath, iS, iE)
    jobName = 'ed{}-{}'.format(iS, iE)
    slurm.submitJob(jobName, cmdLine, nH=12, nM=16)
# os.system(cmdLine)
