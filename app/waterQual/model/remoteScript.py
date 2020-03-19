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

# slurm.submitJobGPU('basinRef','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/trainModel1.py',nH=24)

# slurm.submitJobGPU(
#     'basinAll', 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/trainModel2.py', nH=48, nM=64)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/runCmd.py -D {} -O {}'
nameP = '{}-opt{}'
dLst = ['HBN', 'HBN-30d']
optLst = [1, 2, 3, 4]
for d in dLst:
    for opt in optLst:
        print(cmdP.format(d, opt))
        slurm.submitJobGPU(nameP.format(d, opt), cmdP.format(d, opt), nH=8)
