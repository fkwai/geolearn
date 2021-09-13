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
            'H', 'I', 'J', 'K', 'L', 'M', 'O', 'Q']

subsetLst = ['EcoB{}'.format(x) for x in ecoIdLst]
globalName = '{}-B10'.format(dataName)

hsLst = [16, 64]
caseLst = list()
for subset in subsetLst:
    for hs in hsLst:
        outName = '{}-{}-h{}-B10-gs'.format(dataName, subset, hs)
        caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                        varY=varY, varXC=varXC, varYC=varYC, sd=sd, ed=ed,
                                        hiddenSize=hs, subset=subset, borrowStat=globalName)
        caseLst.append(caseName)
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName))
