

from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataNameLst = ['bsWN5', 'bsDN5', 'brWN5', 'brDN5']

dataName='bsWN5'

dm = dbBasin.DataModelFull(dataName)

varX = dm.varF
varY = ['runoff']+dm.varC
# varY = ['runoff']
varXC = dm.varG
varYC = None
sd = '1982-01-01'
ed = '2009-12-31'


outName = '{}-B10'.format(dataName)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             sd=sd, ed=ed, nEpoch=100,
                             batchSize=[365, 100])

basinFull.trainModel(outName)
