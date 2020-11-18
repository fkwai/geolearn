from .read import *
from .download import *
from hydroDL import kPath
import os
import pandas as pd

fileCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'codeWQ.csv')
codePdf = pd.read_csv(fileCode, dtype=str).set_index('code')
codeLst = list(codePdf.index)
# codeLst.remove('00440')
varC = codeLst
newC = varC.copy()
newC.remove('00440')
newC.remove('00410')
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
    '80154': 'log-norm',
    '00010-R': 'norm',
    '00060-R': 'norm',
    '00094-R': 'norm',
    '00095-R': 'norm',
    '00300-R': 'norm',
    '00301-R': 'norm',
    '00400-R': 'norm',
    '00403-R': 'norm',
    '00405-R': 'norm',
    '00408-R': 'norm',
    '00410-R': 'norm',
    '00440-R': 'norm',
    '00418-R': 'norm',
    '00419-R': 'norm',
    '00530-R': 'norm',
    '00600-R': 'norm',
    '00605-R': 'norm',
    '00618-R': 'norm',
    '00653-R': 'norm',
    '00660-R': 'norm',
    '00665-R': 'norm',
    '00681-R': 'norm',
    '00915-R': 'norm',
    '00925-R': 'norm',
    '00930-R': 'norm',
    '00935-R': 'norm',
    '00940-R': 'norm',
    '00945-R': 'norm',
    '00950-R': 'norm',
    '00955-R': 'norm',
    '39086-R': 'norm',
    '39087-R': 'norm',
    '70303-R': 'norm',
    '71846-R': 'norm',
    '80154-R': 'norm'
}

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
