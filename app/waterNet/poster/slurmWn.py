
from hydroDL.master import slurm


cmdP1 = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterNet/poster/trainWn.py -m WaterNet0119 -d {}'
cmdP2 = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterNet/poster/trainWnTest.py -m WaterNet0630 -d {}'

dataLst = ['QN90ref', 'Q95ref']
for dataName in dataLst:
    slurm.submitJobGPU('WaterNet0119-{}'.format(dataName),
                       cmdP1.format(dataName), nH=24, nM=32)
    slurm.submitJobGPU('WaterNet0630-{}'.format(dataName),
                       cmdP2.format(dataName), nH=24, nM=32)
