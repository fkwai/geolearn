from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)
caseLst = list()

varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']
# ntn variables
for code in varNtnUsgsLst:
    if code == 'comb':
        varYC = usgs.varC
        varXLst = [gridMET.varLst,
                   gridMET.varLst + varNtnLst+['distNTN']]
    else:
        varYC = [code]
        varXLst = [gridMET.varLst,
                   gridMET.varLst + [varNtnLst[varNtnUsgsLst.index(code)], 'distNTN']]
    labelLst = ['plain', 'ntnS']
    subsetLst = ['{}-Y{}'.format(code, x) for x in [1, 2]]
    # wrap up
    # for subset in subsetLst:
    subset = subsetLst[0]
    for k in range(len(varXLst)):
        varX = varXLst[k]
        label = labelLst[k]
        saveName = '{}-{}-{}-{}'.format(dataName, code, label, subset)
        caseName = basins.wrapMaster(
            dataName=dataName, trainName=subset, batchSize=[None, 200],
            outName=saveName, varX=varX, varYC=varYC)
        caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=12)
