
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


t = pd.date_range(start='2019-12-31', end='2020-02-01', freq='D')
v = np.arange(len(t))

df = pd.DataFrame(index=t, columns=['v'], data=v)

offset = pd.offsets.timedelta(days=-6)
df.resample('W-MON', loffset=offset).mean()
v[3::7]
