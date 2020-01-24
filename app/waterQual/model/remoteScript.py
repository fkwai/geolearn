from hydroDL.master import slurm

slurm.submitJobGPU('modelA','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelA.py',nH=20)

slurm.submitJobGPU('modelC','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelC.py',nH=20)