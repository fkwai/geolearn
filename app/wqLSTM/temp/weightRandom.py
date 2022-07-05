from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import numpy as np

dataName = 'N200'
label = 'QFPRT2C'

trainSet = 'rmYr5'
testSet = 'pkYr5'
varX = dbBasin.label2var(label.split('2')[0])
mtdX = dbBasin.io.extractVarMtd(varX)
varY = dbBasin.label2var(label.split('2')[1])
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

DF = dbBasin.DataFrameBasin(dataName)
DM = dbBasin.DataModelBasin(DF, subset=trainSet,
                            varX=varX, varY=varY, varXC=varXC, varYC=varYC)
DM.trans(mtdX=mtdX, mtdXC=mtdXC,
         mtdY=mtdY, mtdYC=mtdYC)
dataTup = DM.getData()
dataTup = basinFull.trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
dataLst = dataTup

matB = ~np.isnan(dataLst[2])
matD = np.any(matB, axis=2)
nD = np.sum(np.any(matB, axis=2))


rho = 365
nbatch = 500
filt = np.ones(rho, dtype=int)
countT = np.apply_along_axis(lambda m: np.convolve(
    m, filt, mode='valid'), axis=0, arr=matD)
wS = np.sum(countT, axis=0)/np.sum(countT)
wT = countT/np.sum(countT, axis=0)

nt, ns = matD.shape
iS = np.random.choice(nt, nbatch, p=wS)
iT = np.zeros(nbatch).astype(int)
for k in range(nbatch):
    iT[k] = np.random.choice(nt-rho, p=wT[:, iS[k]])

pr = np.mean(countT)*500/np.sum(matD)
int(np.ceil(np.log(0.01) / np.log(1 - pr)))


aa = np.convolve(matD[:, k], np.ones(rho, dtype=int), 'valid')
np.sum(a[:, k]-aa)

a = np.random.rand(2, 2)
p = np.random.rand(2, 2)

np.random.choice(a, p=p)
