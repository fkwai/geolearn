from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataNameLst = ['brWN5', 'brDN5']
labelLst = ['QFPRT2C', 'QFPT2C', 'FPRT2QC', 'QT2C']
rhoLst = [365, 10]

for dataName in dataNameLst:
    for label in labelLst:
        for rho in rhoLst:
            varX = dbBasin.label2var(label.split('2')[0])
            varY = dbBasin.label2var(label.split('2')[1])
            varXC = gageII.varLst
            varYC = None
            sd = '1982-01-01'
            ed = '2009-12-31'
            outName = '{}-{}-t{}-B10'.format(dataName, label, rho)
            dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                                         varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                         sd=sd, ed=ed, nEpoch=500,
                                         batchSize=[rho, 1000])
            cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
            slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=32)
