
import statsmodels
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, transform

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
from sklearn.linear_model import LinearRegression

wqData = waterQuality.DataModelWQ('Silica64')
siteNoLst = wqData.siteNoLst
varX = ['00060'] + gridMET.varLst
varY = ['00955']
siteNo = siteNoLst[0]
dfX = waterQuality.readSiteX(siteNo, varX)
dfY = waterQuality.readSiteY(siteNo, varY)

dfY1 = dfY[dfY.index < np.datetime64('2000-01-01')]
dfY2 = dfY[dfY.index >= np.datetime64('2000-01-01')]
dfX1 = dfX[dfX.index < np.datetime64('2000-01-01')]
dfX2 = dfX[dfX.index >= np.datetime64('2000-01-01')]


mod1 = sm.tsa.statespace.SARIMAX(dfY1, exog=dfX1, order=(1, 0, 0))
res1 = mod1.fit(disp=False)
pred1 = res1.get_prediction()
mod2 = sm.tsa.statespace.SARIMAX(dfY2, exog=dfX2, order=(1, 0, 0))
res2 = mod2.filter(res1.params)
pred1 = res1.get_prediction()
pred2 = res2.get_prediction()
dfP1 = pred1.predicted_mean
dfP2 = pred2.predicted_mean
fig, ax = plt.subplots(1, 1)
ax.plot(dfP1, '-b')
ax.plot(dfP2, '-b')
ax.plot(dfY, '*r')
fig.show()


class testMod(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super(testMod, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )

        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                           [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        return [np.std(self.endog)]*3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(testMod, self).update(params, *args, **kwargs)

        # Observation covariance
        self.ssm['obs_cov', 0, 0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]


# Create and fit the model
mod = testMod(dfY)
res = mod.fit()
print(res.summary())
