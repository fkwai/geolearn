import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL.master import default, wrapMaster, runTrain, train


caseLst = ['080305', '080300', '080305+080400+080500', '080305+090200',
           '080305+090300', '080305+090400', '080305+100200', '080305+060200'] +\
    ['090303', '090300', '090303+090401',
     '090303+090402', '090303+100204', '090303+060200'] +\
    ['090401', '090401+090402', '090401+090301', '090401+090203',
     '090401+100105', '090401+080305', '090401+100204']

# init
rootDB = pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
subsetPattern = 'ecoRegionL3_{}_v2f1'
lat, lon = df.getGeo()
fieldLst = ['ecoRegionL'+str(x+1) for x in range(3)]
codeLst = df.getDataConst(fieldLst, doNorm=False, rmNan=False).astype(int)

# create subset
for case in caseLst:
    subsetLst = case.split('+')
    for subset in subsetLst:
        ind = ecoReg_ind(subset, codeLst)
        subsetName = 'ecoRegL3_'+subset+'v2f1'
        # df.subsetInit(subsetPattern.format(subset), ind=ind)

# train
cid = 2
caseLst=['080305+080400+080500']
for case in caseLst:
    subsetLst = [subsetPattern.format(x) for x in case.split('+')]
    outName = subsetPattern.format(case) + '_Forcing'
    varLst = dbCsv.varForcing
    optData = default.update(
        default.optDataSMAP,
        rootDB=pathSMAP['DB_L3_NA'],
        tRange=[20150401, 20160401],
        varT=varLst)
    optData = default.forceUpdate(
        optData, subset=subsetLst)
    optModel = default.optLstm
    optLoss = default.optLossRMSE
    optTrain = default.optTrainSMAP
    out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionL3',
                       outName)
    masterDict = wrapMaster(out, optData, optModel, optLoss,
                            optTrain)
    runTrain(masterDict, cudaID=cid % 3, screen=case)
    # train(masterDict)
    cid = cid+1
