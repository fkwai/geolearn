from hydroDL.master import slurm
from hydroDL import kPath
import os
# slurm.submitJobGPU('modelA','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelA.py',nH=20)

# slurm.submitJobGPU('modelC','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelC.py',nH=20)

# wrap up data
# codePath = os.path.join(kPath.dirCode, 'app',
#                         'waterQual', 'model', 'wrapData.py')
# jobName = 'wrapUpData'
# cmdLine = 'python {}'.format(codePath)
# slurm.submitJob(jobName, cmdLine, nH=1, nM=64)

# TRAIN MODEL

# slurm.submitJobGPU('modelA','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/trainModel1.py',nH=24)

slurm.submitJobGPU(
    'modelAll', 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/trainModel2.py', nH=24, nM=64)
