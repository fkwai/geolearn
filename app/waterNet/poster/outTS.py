

import os
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
from hydroDL.model import waterNetTest, waterNet


# testSet = 'WYA09'
# dataName = 'QN90ref'
# wnName = 'WaterNet0119'

# testSet = 'WYA09'
# dataName = 'Q95ref'
# wnName = 'WaterNet0119'


def fowardWN(wnName, dataName, testSet):
    trainSet = 'WYB09'
    epWN = 500
    if wnName == 'WaterNet0630':
        funcM = getattr(waterNetTest, wnName)
    else:
        funcM = getattr(waterNet, wnName)
    if testSet == 'WYA09':
        testBatch = 100
    elif testSet == 'WYall':
        testBatch = 20

    dirModel = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
    dirOut = r'C:\Users\geofk\work\waterQuality\waterNet\outTemp'
    modelFile = '{}-{}-ep{}'.format(wnName, dataName, epWN)
    DF = dbBasin.DataFrameBasin(dataName)

    # waterNet setup
    varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
    mtdX = ['skip' for k in range(2)] +\
        ['scale' for k in range(2)] +\
        ['norm' for k in range(2)]
    varY = ['runoff']
    mtdY = ['skip']
    varXC = gageII.varLstEx
    mtdXC = ['QT' for var in varXC]
    varYC = None
    mtdYC = dbBasin.io.extractVarMtd(varYC)

    # data
    DM1 = dbBasin.DataModelBasin(
        DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
    DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
    DM2 = dbBasin.DataModelBasin(
        DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
    DM2.borrowStat(DM1)
    dataTup2 = DM2.getData()
    [x, xc, y, yc] = dataTup2
    t = DF.getT(testSet)

    # model waterNet
    nh = 16
    ng = len(varXC)
    ns = len(DF.siteNoLst)
    nr = 5
    model = funcM(nh, len(varXC), nr)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(dirModel, modelFile)))
    model.eval()
    xP = torch.from_numpy(x).float().cuda()
    xcP = torch.from_numpy(xc).float().cuda()
    nt, ns, _ = y.shape
    iS = np.arange(0, ns, testBatch)
    iE = np.append(iS[1:], ns)

    # output time series
    yP = np.ndarray([nt-nr+1, ns])
    [QpP, QsP, QgP] = [np.ndarray([nt-nr+1, ns, nh]) for ii in range(3)]
    [SfP, SsP, SgP] = [np.ndarray([nt, ns, nh]) for ii in range(3)]
    for k in range(len(iS)):
        print('batch {}'.format(k))
        xP = torch.from_numpy(x[:, iS[k]:iE[k], :]).float().cuda()
        xcP = torch.from_numpy(xc[iS[k]:iE[k]]).float().cuda()
        yOut, (QpR, QsR, QgR), (SfT, SsT, SgT) = model(xP, xcP, outStep=True)
        print('done forward')
        yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
        for mT, mN in zip([QpR, QsR, QgR, SfT, SsT, SgT],
                          [QpP, QsP, QgP, SfP, SsP, SgP]):            
            mN[:, iS[k]:iE[k], :] = mT.detach().cpu().numpy()
        print('done detach')

    outName = 'ts{}-{}'.format(modelFile, testSet)
    outFile = os.path.join(dirOut, outName)
    np.savez_compressed(outFile, yP=yP, QpP=QpP, QsP=QsP,
                        QgP=QgP, SfP=SfP, SsP=SsP, SgP=SgP)


# testSet = 'WYA09'
# dataName = 'QN90ref'
# wnName = 'WaterNet0119'

testSet = 'WYA09'
# testSet = 'WYall'
for dataName in ['QN90ref', 'Q95ref']:
    for wnName in ['WaterNet0119', 'WaterNet0630']:
        fowardWN(wnName, dataName, testSet)
