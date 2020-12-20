from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.master import basinFull


varX = gridMET.varLst
varY = ['runoff']
varXC = gageII.lstWaterQuality
varYC = None
dataName = 'Q90'

sd = '1979-01-01'
ed = '2010-01-01'

ecoIdLst = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q']

subsetLst = ['Eco{}'.format(x) for x in ecoIdLst]

dataName = 'Q90ref'
globalName = '{}-B10'.format(dataName)

caseLst = list()
for subset in subsetLst:
    outName = '{}-{}-B10-gs'.format(dataName, subset)
    caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                    varY=varY, varXC=varXC, varYC=varYC, sd=sd, ed=ed,
                                    subset=subset, borrowStat=globalName)
    caseLst.append(caseName)
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName))
