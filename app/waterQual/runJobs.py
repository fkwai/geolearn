from hydroDL.master import slurm
import os
from hydroDL import kPath

""" extract data from gridMET """
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
for var in varLst:

    if kPath.host == 'icme':
        cmd = 'python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var {}'
    elif kPath.host == 'sherlock':
        cmd = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var {}'
    cmdLine = cmd.format(var)
    jobName = var
    slurm.submitJob(jobName, cmdLine, nH=8, nM=16)

# cmdLine = "screen -dmS test bash -c "+\
#     "'srun --exclusive --time 8:0:0 --pty bash;source activate pytorch;" +\
#     "python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 0 -E 10'"
# os.system(cmdLine)
""" 
srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
"""