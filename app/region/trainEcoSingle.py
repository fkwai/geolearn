import numpy as np
from hydroDL import pathSMAP
from hydroDL.master import default, wrapMaster, runTrain, train
import os
from hydroDL.data import dbCsv
import argparse
# train for each cont
subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst = ['Forcing', 'Soilm']

# train for each region combo
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rid', dest='regionId', type=int)
    args = parser.parse_args()
    regionId = args.regionId

# k = [7, 8, 13]
regionId=7
for k in range(len(subsetLst)):
    kc = regionId-1
    if k != kc:
        outName = 'ecoRegion{:02d}{:02d}_v2f1'.format(
            regionId, k+1) + '_Forcing'
        varLst = dbCsv.varForcing
        optData = default.update(
            default.optDataSMAP,
            rootDB=pathSMAP['DB_L3_NA'],
            tRange=[20150401, 20160401],
            varT=varLst)
        optData = default.forceUpdate(
            optData,
            subset=[subsetLst[kc], subsetLst[k]])
        optModel = default.optLstm
        optLoss = default.optLossRMSE
        optTrain = default.optTrainSMAP
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                           outName)
        masterDict = wrapMaster(out, optData, optModel, optLoss,
                                optTrain)
        train(masterDict)

'''
source /home/kxf227/anaconda3/bin/activate
conda activate pytorch

CUDA_VISIBLE_DEVICES=0 python /home/kxf227/work/GitHUB/hydroDL-dev/app/region/trainEcoComb.py --rid 7
CUDA_VISIBLE_DEVICES=1 python /home/kxf227/work/GitHUB/hydroDL-dev/app/region/trainEcoComb.py --rid 8
CUDA_VISIBLE_DEVICES=2 python /home/kxf227/work/GitHUB/hydroDL-dev/app/region/trainEcoComb.py --rid 13
'''
