import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
from hydroDL import kPath
import os
import pandas as pd
import numpy as np

dirPlyn = os.path.join(kPath.dirData, 'isotope', 'Plynlimon')

fileH = os.path.join(dirPlyn, 'stable',
                     'plynlimon_isotopes_and_chloride_7hourly_2007_2009.txt')
fileW = os.path.join(dirPlyn, 'stable',
                     'plynlimon_isotopes_and_chloride_weekly_2004_2009.txt')

dfH = pd.read_csv(fileH, delimiter='\t')
dictRename = {'Date_Time yyyy.mm.dd HH:MM': 'dateTime',
              'Cl mg/l': 'Cl', 'delta_18O': 'O18', 'delta_2H': 'H2',
              'Runoff (mm/15min)': 'Runoff', 'Rainfall (mm)': 'Rainfall',
              'water flux (mm/hr)': 'waterFlux'}
dfH = dfH.rename(columns=dictRename)
dfH['dateTime'] = pd.to_datetime(dfH['dateTime'])
dfH = dfH.set_index('dateTime')
# remove some Cl records
dfH.loc[dfH['Dry deposition affected (yes/no)'] == 'yes', 'Cl'] = np.nan
varLst = ['waterFlux', 'Cl', 'O18', 'H2']
dfP = dfH[dfH['Site'] == 'CR'][varLst]
dfQ = dfH[dfH['Site'] == 'UHF'][varLst]

dfP['Rainfall']/dfP['waterFlux']
dfQ['Runoff']/dfQ['waterFlux']

# plot
fig, axes = plt.subplots(3, 1, sharex=True)
isoVar = 'O18'
axes[0].plot(dfP.index, dfP['waterFlux'], 'b-')
axes[0].plot(dfQ.index, dfQ['waterFlux'], 'g-')
axes[1].plot(dfP.index, dfP['waterFlux'], 'b-')
axes[1].twinx().plot(dfP.index, dfP[isoVar], 'r*')
ind = (dfP['waterFlux'] > 0) & (dfP[isoVar].isna())
axes[1].plot(dfP.index[ind], dfP['waterFlux'][ind], 'k*')
axes[2].plot(dfQ.index, dfQ['waterFlux'], 'b-')
axes[2].twinx().plot(dfQ.index, dfQ[isoVar], 'r*')
fig.show()
