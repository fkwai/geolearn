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

l3Lst = ['080304', '080305', '080401', '080404',
         '080503', '080501', '090402', '090303']
subsetLst = list()
for l3 in l3Lst:
    subsetLst.append('Eco'+l3[:2])
    subsetLst.append('Eco'+l3[:4])
    subsetLst.append('Eco'+l3[:6])
subsetLst = list(set(subsetLst))

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
