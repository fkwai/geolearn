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

outName = '{}-B10'.format(dataName)
caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                varY=varY, varXC=varXC, varYC=varYC,
                                sd=sd, ed=ed)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/model/cmd/basinFull.py -M {}'
slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24, nM=32)
