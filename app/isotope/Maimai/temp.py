import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
from hydroDL import kPath
import os
import pandas as pd

dirMM = os.path.join(kPath.dirData, 'isotope', 'Maimai')
fileMM = os.path.join(dirMM, 'Rainfall_Runoff_Isotope_1987.csv')

df = pd.read_csv(fileMM)
dictRename = {'Date': 'date', 'Rainfall Deut, per mil': 'P_H2',
              'Rainfall EC, [S/m] ': 'P_EC',
              'Rainfall Cl, mg/l': 'P_Cl',
              'Flow Deut, per mil': 'Q_H2',
              'Flow EC, [S/m] ': 'Q_EC',
              'Flow Cl, mg/l': 'Q_Cl'}
df = df.rename(columns=dictRename)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

dfI = df.interpolate()
dfD = df.resample('D').mean()

# plot to check subsample
fig, ax = plt.subplots()
ax.plot(df.index, df['P_H2'], 'b*-')
ax.plot(dfD.index, dfD['P_H2'], 'g*-')
fig.show()

# plot to check subsample
fig, ax = plt.subplots()
ax.plot(dfD.index, dfD['P_H2'], 'b*-')
ax.plot(dfD.index, dfD['Q_H2'], 'g*-')
fig.show()
