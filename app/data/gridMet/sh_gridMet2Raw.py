import os
from hydroDL import kPath
from hydroDL.master import slurm

# extract gridMET to raw data (one file for one variable, multiple sites)
varLst = ["pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr"]
codePath = os.path.join(kPath.dirCode, "app", "data", "gridMet", "gridMetExtract.py")
cmdPtn = "python {} -var {} -syr {} -eyr {} -smask {} -emask {}"
nSite = 9067
syr = 1979
eyr = 2023
iSLst = list(range(0, nSite, 1500))
iELst = iSLst[1:] + [nSite]
for var in varLst:
    for iS, iE in zip(iSLst, iELst):
        cmdLine = cmdPtn.format(codePath, var, syr, eyr, iS, iE)
        jobName = "{}_{}_{}_{}_{}".format(var, iS, iE, syr, eyr)
        slurm.submitJob(jobName, cmdLine, nH=4, nM=64)
