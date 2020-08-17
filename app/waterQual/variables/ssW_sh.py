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
# codeLst = usgs.varC+['comb']
codeLst = ['00300']+varNtnUsgsLst+['comb']
for code in codeLst:
    xLst = [[], ntn.varLst+['distNTN']]
    varXLst = [gridMET.varLst+lst for lst in xLst]
    varYLst = [['00060'], ['00060']]
    labelLst = ['plain', 'ntn']
    if code in varNtnUsgsLst:
        temp = gridMET.varLst+[varNtnLst[varNtnUsgsLst.index(code)], 'distNTN']
        varXLst.append(temp)
        varYLst.append(['00060'])
        labelLst.append('ntnS')
    subsetLst = ['{}-Y{}'.format(code, x) for x in [1, 2]]

    # wrap up
    # for subset in subsetLst:
    subset = subsetLst[0]
    for k in range(len(varXLst)):
        varX = varXLst[k]
        varY = varYLst[k]
        label = labelLst[k]
        if code == 'comb':
            varYC = usgs.varC
        else:
            varYC = [code]
        saveName = '{}-{}-{}-{}'.format(dataName, code, label, subset)
        caseName = basins.wrapMaster(
            dataName=dataName, trainName=subset, batchSize=[None, 200],
            outName=saveName, varX=varX, varY=varY, varYC=varYC)
        caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
