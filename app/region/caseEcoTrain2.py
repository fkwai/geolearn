import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL.master import default, wrapMaster, runTrain, train


caseLst = ['080305', '090301', '090303',
           '090401', '090402', '100105', '100204']

# caseLst = ['090301', '090303', '090401', '090402', '100105', '100204']

# init
rootDB = pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
subsetPattern = 'ecoReg_{}_L{}_v2f1'
lat, lon = df.getGeo()

# train
cid = 2
for case in caseLst:
    for level in [3, 2, 1, 0]:
        subset = subsetPattern.format(case, level)
        outName = subsetPattern.format(case, level) + '_Forcing'
        varLst = dbCsv.varForcing
        optData = default.update(
            default.optDataSMAP,
            rootDB=pathSMAP['DB_L3_NA'],
            tRange=[20150401, 20160401],
            varT=varLst)
        optData = default.forceUpdate(
            optData, subset=subset)
        optModel = default.optLstm
        optLoss = default.optLossRMSE
        optTrain = default.optTrainSMAP
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionCase',
                           outName)
        masterDict = wrapMaster(out, optData, optModel, optLoss,
                                optTrain)
        runTrain(masterDict, cudaID=cid % 3, screen=case)
        # train(masterDict)
    cid = cid+1
