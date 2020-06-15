from astropy.timeseries import LombScargle
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

import scipy.signal as signal

wqData = waterQuality.DataModelWQ('Silica64')
siteNoLst = wqData.siteNoLst

for siteNo in siteNoLst:
		print(siteNo)
		dfObs = waterQuality.readSiteY(siteNo, ['00955'])
		# rm outlier
		df = dfObs[dfObs['00955'].notna().values]
		y = df['00955'].values
		yV = y[y < np.percentile(y, 99)]
		yV = yV[yV > np.percentile(y, 1)]
		ul = np.mean(yV)+np.std(yV)*5
		dfObs[dfObs['00955'] > ul] = np.nan
		# fourier
		df = dfObs[dfObs.notna().values]
		tt = dfObs.index.values
		xx = (tt.astype('datetime64[D]') -
			np.datetime64('1979-01-01')).astype(np.float)
		t = df.index.values
		x = (t.astype('datetime64[D]') -
			np.datetime64('1979-01-01')).astype(np.float)
		y = df['00955'].values
		y = y-np.nanmean(y)
		nt = len(xx)
		# nt = 1000
		# freq = 1/np.linspace(2, nt, nt)
		# freq = np.arange(1, nt)/nt
		freq = np.fft.fftfreq(nt)[1:]

		ls = LombScargle(x, y)
		power = ls.power(freq)
		xx = (dfObs.index.values.astype('datetime64[D]') -
			np.datetime64('1979-01-01')).astype(np.float)

		ym = np.zeros([len(freq), len(xx)])
		for k, f in enumerate(freq):
			ym[k, :] = ls.model(xx, f)
		folder = r'C:\Users\geofk\work\waterQuality\tempData\LS'
		np.save(os.path.join(folder, siteNo+'-full'), ym)
