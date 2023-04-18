from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = dbBasin.DataFrameBasin('NY5')

df2 = dbBasin.DataFrameBasin('G200')

siteNoLst = df2.siteNoLst

ind2 = 50
ind1 = df1.siteNoLst.index(df2.siteNoLst[ind2])

# plot q - runoff calculated wrong
unitConv = 0.3048**3 * 24 * 60 * 60 / 1000
area=df1.g[ind1,0]

fig, ax = plt.subplots(1, 1)
ax.plot(df1.t, df1.q[:, ind1, 0]/area*unitConv, '-r')
ax.plot(df1.t, df1.q[:, ind1, 1], '-b')
# ax.plot(df2.t, df2.q[:, ind2, 1], '-b')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(df1.t, df1.q[:, ind1, 0], '-r')
ax.plot(df2.t, df2.q[:, ind2, 0], '-b')
fig.show()


# plot pr - almost identical slightly different boundary
fig, ax = plt.subplots(1, 1)
ax.plot(df1.t, df1.f[:, ind1, 0], '-r')
ax.plot(df2.t, df2.f[:, ind2, 0], '-b')
fig.show()


# plot code - identical
ind2 = 10
ind1 = df1.siteNoLst.index(df2.siteNoLst[ind2])
code='00915'
ic1=df1.varC.index(code)
ic2=df2.varC.index(code)
fig, ax = plt.subplots(1, 1)
ax.plot(df1.t, df1.c[:, ind1, ic1], '*r')
ax.plot(df2.t, df2.c[:, ind2, ic2], '*b')
fig.show()
