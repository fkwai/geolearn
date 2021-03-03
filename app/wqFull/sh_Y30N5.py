from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.master import basinFull


varX = gridMET.varLst
varY = ['runoff']+usgs.newC
varXC = gageII.lstWaterQuality
varYC = None
dataName = 'sbTest'

sd = '1979-01-01'
ed = '2010-01-01'

outName = '{}-B10'.format(dataName)
master = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                varY=varY, varXC=varXC, varYC=varYC,
                                sd=sd, ed=ed)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=32)
