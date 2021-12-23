from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.master import basinFull


varX = gridMET.varLst
varY = ['runoff']
varXC = ['DRAIN_SQKM', 'SNOW_PCT_PRECIP', 'GEOL_REEDBUSH_DOM',
         'STREAMS_KM_SQ_KM', 'PCT_1ST_ORDER', 'BFI_AVE', 'CONTACT',
         'FORESTNLCD06', 'PLANTNLCD06', 'NUTR_BAS_DOM',
         'HLR_BAS_DOM_100M', 'ELEV_MEAN_M_BASIN',
         'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE', 'SLOPE_PCT']
varYC = None
dataName = 'Q90ref-rmEco'

sd = '1979-01-01'
ed = '2010-01-01'

outName = '{}-B10'.format(dataName)
caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                varY=varY, varXC=varXC, varYC=varYC,
                                sd=sd, ed=ed)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24, nM=32)
