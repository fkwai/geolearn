# Global Sinusoidal Seasonality

from hydroDL import kPath, utils
import pandas as pd
import os
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
dirGNIP = os.path.join(kPath.dirData, 'isotope', 'GNIP')
dfSD = pd.read_csv(os.path.join(dirGNIP, 'siteD.csv'))
dfSM = pd.read_csv(os.path.join(dirGNIP, 'siteM.csv'))

df = dfSM
lat = df['Latitude'].values
lon = df['Longitude'].values
count1 = df['O18-count'].values
count2 = df['H2-count'].values
figM = plt.figure(figsize=(12, 5))
gsM = gridspec.GridSpec(2, 1)
axM1 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, count1)
axM1.set_title('Count of O18')
axM2 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, count2)
axM2.set_title('Count of H2')
figM.show()