import datetime as dt
import numpy as np
import pandas as pd

rd = np.datetime64('2000-01-01', 'D')  # reference date


def t2dt(t, hr=False):
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception('hydroDL.utils.t2dt failed')
    return tOut


def tRange2Array(tRange, *, step=np.timedelta64(1, 'D')):
    sd = t2dt(tRange[0])
    ed = t2dt(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def intersect(tLst1, tLst2):
    # numpy>1.14
    # C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)

    # numpy<1.14
    C = np.intersect1d(tLst1, tLst2)
    ind1 = np.where(np.isin(tLst1, C))[0]
    ind2 = np.where(np.isin(tLst2, C))[0]

    return ind1, ind2


def datePdf(df, field='date'):
    df['date'] = pd.to_datetime(df[field])
    df = df.set_index('date')
    return df


def date2num(t):
    return (t.astype('datetime64[D]')-rd).astype(int)


def num2date(tn):
    return rd+tn
