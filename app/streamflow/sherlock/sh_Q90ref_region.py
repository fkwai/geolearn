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

# subsetLst = [
#     '09', '0903', '090303',
#     '0904', '090402',
#     '08', '0803', '080305',
# ]
subsetLst = [
    '0804', '080401', '080304',
    '0805', '080503'
]
caseLst = list()
for subset in subsetLst:
    outName = '{}-{}-B10'.format(dataName, subset)
    caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                    varY=varY, varXC=varXC, varYC=varYC,
                                    sd=sd, ed=ed, subset=subset)
    caseLst.append(caseName)
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24, nM=32)
