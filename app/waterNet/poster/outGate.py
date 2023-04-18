

from hydroDL.model.waterNet import convTS, sepPar
import os
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
from hydroDL.model import waterNetTest, waterNet


# testSet = 'WYA09'
# dataName = 'QN90ref'
# wnName = 'WaterNet0119'

dataName = 'Q95ref'
wnName = 'WaterNet0119'

def getGate(wnName, dataName):
    trainSet = 'WYB09'
    epWN = 500
    if wnName == 'WaterNet0630':
        funcM = getattr(waterNetTest, wnName)
    else:
        funcM = getattr(waterNet, wnName)
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
    dataTup1 = DM1.getData()

    # model waterNet
    nh = 16
    ng = len(varXC)
    ns = len(DF.siteNoLst)
    nr = 5
    model = funcM(nh, len(varXC), nr)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(dirModel, modelFile)))
    model.eval()


    [xN, xcN, yN, ycN] = dataTup1
    # constant parameters
    xc = torch.from_numpy(xcN).float().cuda()
    w = model.fcW(xc)
    [kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, model.wLst)
    if wnName == 'WaterNet0119':
        gL = gL**2
        kg = kg/10
        ga = torch.softmax(ga, dim=-1)
    elif wnName == 'WaterNet0630':
        gL = gL**2
        qb = qb + 1e-8
        ga = torch.softmax(ga, dim=-1)
    dictOut = dict()
    lat, lon = DF.getGeo()
    dictOut['lat'] = lat
    dictOut['lon'] = lon
    for v, vName in zip([kp, ks, kg, gp, gL, qb, ga],
                        ['kp', 'ks', 'kg', 'gp', 'gL', 'qb', 'ga']):
        dictOut[vName] = v.detach().cpu().numpy()
    outName = 'gate{}'.format(modelFile)
    outFile = os.path.join(dirOut, outName)
    np.savez_compressed(outFile, **dictOut)


# testSet = 'WYall'
for dataName in ['QN90ref', 'Q95ref']:
    for wnName in ['WaterNet0119', 'WaterNet0630']:
        getGate(wnName, dataName)
