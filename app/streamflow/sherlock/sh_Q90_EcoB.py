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

l3Lst = ['080304',
         '050301',
         '080401',
         '090203',
         '080305',
         '080203',
         '080503',
         '090402',
         '080301',
         '080107',
         '080204',
         '080402']

subsetLst = list()
for l3 in l3Lst:
    subsetLst.append('EcoB'+l3[:2])
    subsetLst.append('EcoB'+l3[:4])
    subsetLst.append('EcoB'+l3[:6])
subsetLst = list(set(subsetLst))

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
