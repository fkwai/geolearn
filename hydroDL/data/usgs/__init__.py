from .read import *
from .download import *
from hydroDL import kPath
import os
import pandas as pd

fileCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'codeWQ.csv')
codePdf = pd.read_csv(fileCode, dtype=str).set_index('code')
codeLst = list(codePdf.index)
varC = codeLst
varQ = ['00060']

dictStat = {
    'runoff': 'log-norm',
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
