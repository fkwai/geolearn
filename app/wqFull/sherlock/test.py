from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

if __name__ == '__main__':
    dataNameLst = ['G200Norm', 'G400Norm']
    for dataName in dataNameLst:
        outName = dataName
        DF = dbBasin.DataFrameBasin(dataName)
        testSet = 'all'
        try:
            yP, ycP = basinFull.testModel(
                outName, DF=DF, testSet=testSet, ep=200, reTest=True)
            print('tested {}'.format(outName), flush=True)
        except:
            print('skiped {}'.format(outName), flush=True)
