from .read import *
from .download import *
from hydroDL import kPath
import os
import pandas as pd

fileCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'codeWQ.csv')
codePdf = pd.read_csv(fileCode, dtype=str).set_index('code')
codeLst = list(codePdf.index)

fileSampleFull = os.path.join(kPath.dirData, 'USGS',
                              'inventory', 'usgsSampleCodeFull.csv')
samplePdfFull = pd.read_csv(fileSampleFull, dtype=str).set_index('parm_cd')
sampleFull = list(samplePdfFull.index)
sampleFull.remove('00060')

varC = ['00010', '00095', '00300', '00400', '00405',
        '00600', '00605', '00618', '00660', '00665',
        '00681', '00915', '00925', '00930', '00935',
        '00940', '00945', '00955', '71846', '80154']

codeIso = ['82085', '82082']

varQ = ['00060', 'runoff']
chemLst = ['00300', '00405', '00410', '00440', '00600',
           '00605', '00618', '00660', '00665', '71846',
           '00915', '00925', '00930', '00935', '00940',
           '00945', '00955', '00950', '80154', '00681']

# normalization method
dictStat = {
    'runoff': 'log-norm',
    'qPredY1': 'log-norm',
    'qPredY2': 'log-norm',
    '00010': 'norm',
    '00060': 'log-norm',
    '00094': 'log-norm',
    '00095': 'log-norm',
    '00300': 'norm',
    '00301': 'stan',
    '00400': 'norm',
    '00403': 'norm',
    '00405': 'log-norm',
    '00408': 'stan',
    '00410': 'log-norm',
    '00440': 'log-norm',
    '00418': 'log-stan',
    '00419': 'log-stan',
    '00530': 'log-norm',
    '00600': 'log-norm',
    '00605': 'log-norm',
    '00618': 'log-norm',
    '00653': 'norm',
    '00660': 'log-norm',
    '00665': 'log-norm',
    '00681': 'log-norm',
    '00915': 'log-norm',
    '00925': 'log-norm',
    '00930': 'log-norm',
    '00935': 'log-norm',
    '00940': 'log-norm',
    '00945': 'log-norm',
    '00950': 'log-norm',
    '00955': 'log-norm',
    '39086': 'log-norm',
    '39087': 'log-norm',
    '70303': 'log-norm',
    '71846': 'log-norm',
    '80154': 'log-norm'
}

# update stat mtd for mean, std and norm
dictStatApp = dict()
for code in varC:
    dictStatApp[code+'-N'] = 'skip'
    if code in ['00010', '00300', '00400']:
        dictStatApp[code+'-M'] = 'norm'
        dictStatApp[code+'-S'] = 'norm'
    else:
        dictStatApp[code+'-M'] = 'log-norm'
        dictStatApp[code+'-S'] = 'log-norm'
dictStat.update(dictStatApp)

# code of remarks
dfFlagC = pd.DataFrame([
    [0, 'x', 'No flags'],
    [1, 'X', 'Averaged from no flag'],
    [2, '<', 'less than'],
    [3, '>', 'greater than'],
    [4, 'A', 'average'],
    [5, 'E', 'estimated'],
    [6, 'M', 'presence verified but not quantified'],
    [7, 'N', 'presumptive evidence of presence'],
    [8, 'R', 'radchem non-detect, below ssLc'],
    [9, 'S', 'most probable value'],
    [10, 'U', 'analyzed for but not detected'],
    [11, 'V', 'value affected by contamination']],
    columns=['code', 'label', 'description']).set_index('code')

dfFlagQ = pd.DataFrame([
    [0, '0', 'No flags'],
    [1, '<', 'The Value is known to be less than reported value'],
    [2, '>', 'The value is known to be greater than reported value'],
    [3, 'e', 'The value has been edited or estimated by USGS personnel'],
    [4, 'R', 'Records for these data have been revised'],
    [5, 'A', 'Approved for publication -- Processing and review completed'],
    [6, 'P', 'Provisional data subject to revision']],
    columns=['code', 'label', 'description']).set_index('code')
