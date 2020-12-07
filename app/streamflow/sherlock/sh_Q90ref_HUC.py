from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.master import basinFull


varX = gridMET.varLst
varY = ['runoff']
varXC = gageII.lstWaterQuality
varYC = None
sd = '1979-01-01'
ed = '2010-01-01'
dataName = 'Q90ref'
globalName = '{}-B10'.format(dataName)

subsetLst = ['HUC{:02d}'.format(k+1) for k in range(18)]
caseLst = list()
for subset in subsetLst:
    outName = '{}-{}-B10'.format(dataName, subset)
    caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                    varY=varY, varXC=varXC, varYC=varYC, sd=sd, ed=ed,
                                    subset=subset)
    caseLst.append(caseName)

    outName = '{}-{}-B10-gs'.format(dataName, subset)
    caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                    varY=varY, varXC=varXC, varYC=varYC, sd=sd, ed=ed,
                                    subset=subset, borrowStat=globalName)
    caseLst.append(caseName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName))

# basinFull.trainModel(caseLst[-1])
