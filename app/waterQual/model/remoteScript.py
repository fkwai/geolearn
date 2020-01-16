from hydroDL.master import slurm

slurm.submitJobGPU('modelA','python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/modelA.py',nH=10)