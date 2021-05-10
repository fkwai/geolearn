from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

if __name__ == '__main__':
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
                dm = dbBasin.DataModelFull(dataName)
                testSet = 'all'
                try:
                    yP, ycP = basinFull.testModel(
                        outName, DM=dm, batchSize=20, testSet=testSet, ep=100)
                    print('tested {}'.format(outName), flush=True)
                except:
                    print('skiped {}'.format(outName), flush=True)
