import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

import requests
from io import BytesIO
from zipfile import ZipFile

# Download the dataset
dk = requests.get('http://www.ssfpack.com/files/DK-data.zip').content
f = BytesIO(dk)
zipped = ZipFile(f)
df = pd.read_table(
    BytesIO(zipped.read('internet.dat')),
    skiprows=1, header=None, sep='\s+', engine='python',
    names=['internet', 'dinternet']
)

# Get the basic series
dta_full = df.dinternet[1:].values
dta_miss = dta_full.copy()

# Remove datapoints
missing = np.r_[6, 16, 26, 36, 46, 56, 66, 72, 73, 74, 75, 76, 86, 96]-1
dta_miss[missing] = np.nan


aic_full = pd.DataFrame(np.zeros((6, 6), dtype=float))
aic_miss = pd.DataFrame(np.zeros((6, 6), dtype=float))

warnings.simplefilter('ignore')

# Iterate over all ARMA(p,q) models with p,q in [0,6]
for p in range(6):
    for q in range(6):
        if p == 0 and q == 0:
            continue

        # Estimate the model with no missing datapoints
        mod = sm.tsa.statespace.SARIMAX(
            dta_full, order=(p, 0, q), enforce_invertibility=False)
        try:
            res = mod.fit(disp=False)
            aic_full.iloc[p, q] = res.aic
        except:
            aic_full.iloc[p, q] = np.nan

        # Estimate the model with missing datapoints
        mod = sm.tsa.statespace.SARIMAX(
            dta_miss, order=(p, 0, q), enforce_invertibility=False)
        try:
            res = mod.fit(disp=False)
            aic_miss.iloc[p, q] = res.aic
        except:
            aic_miss.iloc[p, q] = np.nan

# Statespace
mod = sm.tsa.statespace.SARIMAX(dta_miss, order=(1,0,1))
res = mod.fit(disp=False)
print(res.summary())


# In-sample one-step-ahead predictions, and out-of-sample forecasts
nforecast = 20
predict = res.get_prediction(end=mod.nobs + nforecast)
idx = np.arange(len(predict.predicted_mean))
predict_ci = predict.conf_int(alpha=0.5)

# Graph
fig, ax = plt.subplots(figsize=(12,6))
ax.xaxis.grid()
ax.plot(dta_miss, 'k.')

# Plot
ax.plot(idx[:-nforecast], predict.predicted_mean[:-nforecast], 'gray')
ax.plot(idx[-nforecast:], predict.predicted_mean[-nforecast:], 'k--', linestyle='--', linewidth=2)
ax.fill_between(idx, predict_ci[:, 0], predict_ci[:, 1], alpha=0.15)

ax.set(title='Figure 8.9 - Internet series')