import os
import numpy as np
import pandas as pd
from hydroDL import kPath

varLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
flagLst = ['flagCa', 'flagMg', 'flagK', 'flagNa', 'flagNH4',
           'flagNO3', 'flagCl', 'flagSO4', 'valcode', 'invalcode']

dictStat = dict(ph='norm', Conduc='norm', Ca='norm', Mg='norm', K='norm',
                Na='norm', NH4='norm', NO3='norm', Cl='norm', SO4='norm')

