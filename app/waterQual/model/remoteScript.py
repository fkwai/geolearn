from hydroDL.master import slurm
from hydroDL import kPath
import os
# slurm.submitJobGPU('modelA','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelA.py',nH=20)

# slurm.submitJobGPU('modelC','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelC.py',nH=20)

# convert from raw to output
codePath = os.path.join(kPath.dirCode, 'app',
                        'waterQual', 'model', 'wrapData.py')
jobName = 'wrapUpData'
cmdLine = 'python {}'.format(codePath)
slurm.submitJob(jobName, cmdLine, nH=2, nM=64)
