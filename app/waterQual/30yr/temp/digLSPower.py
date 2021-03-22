from hydroDL.app import waterQuality
from hydroDL.data import usgs
from astropy.timeseries import LombScargle
import numpy as np
from hydroDL import kPath, utils
from sklearn.linear_model import LinearRegression

# siteNo = '12340000'
siteNo = '401733105392404'
codeLst = sorted(usgs.newC)

df = waterQuality.readSiteTS(
    siteNo, varLst=['00060']+codeLst, freq='W', rmFlag=True)

code = '00915'
# ls power
t = np.arange(len(df))*7
y = df[code]
tt, yy = utils.rmNan([t, y], returnInd=False)
p = LombScargle(tt, yy).power(1/365)

yr = df.index.year.values
t = yr+df.index.dayofyear.values/365
sinT = np.sin(2*np.pi*t)
cosT = np.cos(2*np.pi*t)
df['sinT'] = np.sin(2*np.pi*t)
df['cosT'] = np.cos(2*np.pi*t)

x = df[['sinT', 'cosT']].values
y = df[code].values
[xx, yy], iv = utils.rmNan([x, y])
lrModel = LinearRegression()
lrModel = lrModel.fit(xx, yy)
y1 = lrModel.predict(df[['sinT', 'cosT']].values)

yy, pp = utils.rmNan([y, y1], returnInd=False)
np.corrcoef(yy, pp)[0]**2
