import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
from hydroDL import kPath
import os
import pandas as pd

dirPlyn = os.path.join(kPath.dirData, 'isotope', 'Plynlimon')

fileHF = os.path.join(dirPlyn, 'highFreq0709', 'data',
                      'PlynlimonHighFrequencyHydrochemistry.csv')
file1 = os.path.join(dirPlyn, 'hydroChem8311', 'data',
                     'PlynlimonResearchCatchmentHydrochemistryData.csv')
file2 = os.path.join(dirPlyn, 'hydroChem1116', 'data',
                     'Plynlimon_hydrochemistry_2011_2016.csv')
file3 = os.path.join(dirPlyn, 'hydroChem1619', 'data',
                     'Plynlimon_hydrochemistry_2016_2019.csv')
dfHF = pd.read_csv(fileHF)
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

cHF = dfHF.columns.tolist()
c1 = df1.columns.tolist()
c2 = df2.columns.tolist()
c3 = df3.columns.tolist()

df1['SITE NAME'].unique()
df2['SITE_NAME'].unique()
df3['SITE_NAME'].unique()
dfHF['Site'].unique()

