import ee
import pandas as pd

# ee.Authenticate()

ee.Initialize()

col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')

lat = 40.0
lon = -100.0
geometry = ee.Geometry.Point([lon, lat])
image = col.filterDate('2021-05-01', '2021-06-01')
a = col.select('SR_B2').filterDate('2021-05-01', '2021-06-01')
a.getInfo()


col = ee.ImageCollection('MODIS/061/MCD43A4')
proj = col.mosaic().projection()
grid = col.geometry().coveringGrid(proj)
a=col.geometry()
b=a.coveringGrid(proj)
b.getInfo()
grid.getInfo()
t=col.mosaic().reduceRegion(reducer=ee.Reducer.mean(), geometry=grid)

t.getInfo()
def record2df(record):
    df = pd.DataFrame.from_records(record[1:], columns=record[0])
    # df.drop('id', axis=1, inplace=True)
    df.time = df.time / 1000
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time'] = df['time'].dt.floor('S')
    # df.rename(columns={'time': 'date'}, inplace=True)
    df = df.sort_values(by='time')
    return df


info = col.getRegion(geometry, 1).getInfo()
df = record2df(info)
df.to_csv('test1.csv', index=False)

info = col.getRegion(geometry, 5).getInfo()
df = record2df(info)
df.to_csv('test5.csv', index=False)

info = col.getRegion(geometry, 30).getInfo()
df = record2df(info)
df.to_csv('test30.csv', index=False)

info = col.getRegion(geometry, 500).getInfo()
df = record2df(info)
df.to_csv('test500.csv', index=False)
