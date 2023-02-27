from hydroDL.master import slurm

dataName = 'rmTK-B200'
# trainLst = ['B15']+['rmYr5b{}'.format(k) for k in range(5)]+['rmRT5b{}'.format(k) for k in range(5)]

trainLst = 'rmYr5b0'

for trainSet in trainLst:
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/WRTDS.py -D {} -T {}'
    slurm.submitJob(trainSet, cmdP.format(dataName, trainSet), nH=24, nM=64)
