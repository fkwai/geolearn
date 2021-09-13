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
