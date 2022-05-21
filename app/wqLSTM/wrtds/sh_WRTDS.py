from hydroDL.master import slurm

dataName = 'G200'
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
for trainSet in trainLst:
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/WRTDS.py -D {} -T {}'
    slurm.submitJob(trainSet, cmdP.format(dataName, trainSet), nH=24, nM=64)
