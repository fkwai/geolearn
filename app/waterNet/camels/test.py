from hydroDL import kPath
from hydroDL.data import dbBasin, camels
from hydroDL.master import basinFull, slurm


varX = camels.varF
mtdX = camels.extractVarMtd(varX)
varY = ['runoff']
mtdY = camels.extractVarMtd(varY)
varXC = camels.varG
mtdXC = camels.extractVarMtd(varXC)
varYC = None
mtdYC = None

trainSet = 'B05'
testSet = 'A05'
dataLst = ['camelsN', 'camelsD', 'camelsM']
rho = 365
# for dataName in dataLst:
dataName = 'camelsN'
DF = dbBasin.DataFrameBasin(dataName)
DF.readSubset('WY8095')
