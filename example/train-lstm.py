from hydroDL import pathSMAP
from hydroDL.master import default, wrapMaster, train
import os

cDir = os.path.dirname(os.path.abspath(__file__))

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB=os.path.join(cDir, 'data'),
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
optModel = default.optLstm   
optLoss = default.optLossRMSE
optTrain = default.update(default.optTrainSMAP, nEpoch=100)
out = os.path.join(cDir, 'output', 'CONUSv4f1')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
train(masterDict)
