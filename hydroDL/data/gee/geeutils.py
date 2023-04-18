import ee
import pandas as pd


def bb2ee(bb):
    [y1, x1, y2, x2] = bb
    rect = ee.Geometry.Rectangle([x1, y1, x2, y2])
    return rect


def t2ee(t):
    tt = ee.Date.fromYMD(t.year, t.month, t.day)
    return tt


def record2df(record):
    df = pd.DataFrame.from_records(record[1:], columns=record[0])
    # df.drop('id', axis=1, inplace=True)
    df.time = df.time / 1000
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time'] = df['time'].dt.floor('S')
    # df.rename(columns={'time': 'date'}, inplace=True)
    df = df.sort_values(by='time')
    return df
