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

l2Lst = ['0801', '0802', '0803', '0804', '0805', '0902', '0903', '0904']

subsetLst = list()
for l2 in l2Lst:
    subsetLst.append('EcoB'+l2[:2])
    subsetLst.append('EcoB'+l2[:4])
subsetLst = list(set(subsetLst))

globalName = '{}-B10'.format(dataName)

caseLst = list()
hsLst = [16, 64]
for subset in subsetLst:
    for hs in hsLst:
        outName = '{}-{}-h{}-B10-gs'.format(dataName, subset, hs)
        caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                        varY=varY, varXC=varXC, varYC=varYC, sd=sd, ed=ed,
                                        hiddenSize=hs, subset=subset, borrowStat=globalName)
        caseLst.append(caseName)
cmdP = 'python /home/users/kuaifang/GitHUB/regional/hydroDL/master/cmd/basinFull.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName))
