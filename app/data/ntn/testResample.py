
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


t = pd.date_range(start='2019-12-31', end='2020-02-1', freq='D')
v = np.arange(len(t)).astype(float)
v[3:16] = np.nan

df = pd.DataFrame(index=t, columns=['v'], data=v)

df.interpolate(limit=5, limit_direction='both', limit_area='inside')

df.resample('W-TUE').mean()


# offset = pd.offsets.timedelta(days=-6)
# df.resample('W-MON', loffset=offset).mean()


df.v.isnull().astype(int).groupby(df.v.notnull().astype(
    int).cumsum()).cumsum()

dfI = df.isnull().astype(int)
dfI.cumsum()
