from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.master import basinFull


varX = gridMET.varLst
varY = ['runoff']
varXC = gageII.lstWaterQuality
varYC = None
dataName = 'Q90ref'

sd = '1979-01-01'
ed = '2010-01-01'

hsLst = [16, 64]
caseLst = list()
for hs in hsLst:
    outName = '{}-h{}-B10'.format(dataName, hs)
    caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                    varY=varY, varXC=varXC, varYC=varYC,
                                    hiddenSize=hs, sd=sd, ed=ed)
    caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/regional/hydroDL/master/cmd/basinFull.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24, nM=32)
