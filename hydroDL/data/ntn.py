import os
import numpy as np
import pandas as pd
from hydroDL import kPath

varLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
flagLst = ['flagCa', 'flagMg', 'flagK', 'flagNa', 'flagNH4',
           'flagNO3', 'flagCl', 'flagSO4', 'valcode', 'invalcode']

dictStat = dict(ph='norm', Conduc='norm', Ca='log-norm', Mg='log-norm',
                K='log-norm', Na='log-norm', NH4='log-norm', NO3='log-norm',
                Cl='log-norm', SO4='log-norm', distNTN='norm')
