from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)
caseLst = list()

varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']

code = '00400'

# xLst = [[], ntn.varLst+['distNTN'], ['00060'],
#         ['00060']+ntn.varLst+['distNTN']]
# varXLst = [gridMET.varLst+lst for lst in xLst]
# varYLst = [['00060'], ['00060'], None, None]
# labelLst = ['plain', 'ntn', 'q', 'ntnq']

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
for subset in subsetLst:
    for k in range(len(varXLst)):
        varX = varXLst[k]
        varY = varXLst[k]
        label = labelLst[k]
        if code == 'comb':
            varYC = usgs.varC
        else:
            varYC = [code]
        saveName = '{}-{}-{}'.format(dataName, label, subset)
        caseName = basins.wrapMaster(
            dataName=dataName, trainName=subset, batchSize=[None, 200],
            nEpoch=nEp, outName=saveName, varX=varX, varY=varY, varYC=varYC)
        caseLst.append(caseName)


for caseName in caseLst:
    basins.trainModelTS(caseName)
