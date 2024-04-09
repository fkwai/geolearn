import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
import json
from hydroDL.post import axplot, figplot, mapplot


data_file='/home/kuai/GitHUB/lfmc_from_sar/input_data/lstm_input_data_pure+all_same_28_may_2019_res_SM_gap_3M'
df1= pd.read_pickle(data_file)
siteNameLst1=df1['site'].unique().tolist()
siteNameLst1.remove('Panter')
siteNameLst1.append('Panther')

# df1.columns.tolist()
aa=df1['date'].unique()
aa.sort()


import os
from hydroDL import kPath

outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabAll = pd.read_csv(outFile)
tabAll.drop(tabAll[tabAll['GACC']=='AICC'].index)
tabAll.drop(tabAll[tabAll['GACC']=='EACC'].index)
tabAll['Date'] = pd.to_datetime(tabAll['Date'], format='%Y-%m-%d')
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)

# find sites that are of single fuel
cntFuel = tabAll.groupby(['siteId'])['Fuel'].nunique().sort_values()
cntSample= tabAll.groupby(['siteId']).size()
cntSample.name='count'
siteSingle = cntFuel[cntFuel == 1].index.tolist()
tabCrdS=tabCrd[tabCrd.index.isin(siteSingle)]
tabCrdS=tabCrdS.join(cntSample)

# remove sites inside Krishna's 150
tabCrdS1=tabCrdS[tabCrdS['Site'].isin(siteNameLst1)]
tabCrdS2=tabCrdS[~tabCrdS['Site'].isin(siteNameLst1)]

df2=tabCrdS2[tabCrdS['count']>60]
lat=df2['lat'].values
lon=df2['lon'].values
count=df2['count'].values

def funcM():
    figM = plt.figure(figsize=(8, 4))
    gsM = gridspec.GridSpec(1, 1)
    # axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, count, s=16, cb=True,vRange=[0,50])    
    axM = mapplot.mapPoint(figM, gsM[0, 0],lat,lon,count, s=16, cb=True,vRange=[0,100])    
    figP,axP=plt.subplots(1,1)    
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteId = df2.index[iP]
    dfTemp=tabAll[tabAll['siteId']==siteId]
    sLst=dfTemp['Fuel'].unique().tolist()
    for s in sLst:
        temp=dfTemp[dfTemp['Fuel']==s]
        t=temp['Date'].values
        v=temp['Percent'].values
        axP.plot(t,v,'-*',label=s)   
    axP.legend()

figM, figP = figplot.clickMap(funcM, funcP)


## land cover
from osgeo import ogr, gdal
lat=tabCrdS['lat'].values
lon=tabCrdS['lon'].values
fileName = '/mnt/sda/dataRaw/NLCD/2016/nlcd_2016.tif'
matLC = np.zeros([len(lat), 9])
ds = gdal.Open(fileName)
gt = ds.GetGeoTransform()
data = ds.GetRasterBand(1).ReadAsArray()
pxA = ((lon - gt[0]) / gt[1]).astype(int)  # x pixel
pyA = ((lat - gt[3]) / gt[5]).astype(int)  # y pixel
for kk, (px, py) in enumerate(zip(pxA, pyA)):
    n = 5
    temp = data[px - n : px + n + 1, py - n : py + n + 1]
    temp = np.floor(temp / 10)
    v = np.zeros(9)
    for k in range(9):
        v[k] = np.sum(temp == k + 1)
    v = v / (n * 2 + 1) ** 2
    matLC[kk, :] = v

cols=['LC{}'.format(x) for x in range(1,10)]
tabLC=pd.DataFrame(index=tabCrdS.index,columns=cols,value=matLC)
tabLC.loc[:]=matLC

tabCrdS=tabCrdS.join(tabLC)

tabCrdS1=tabCrdS[tabCrdS['Site'].isin(siteNameLst1)]
tabCrdS2=tabCrdS[~tabCrdS['Site'].isin(siteNameLst1)]
df2=tabCrdS2[tabCrdS['count']>60]


tabTemp=df2
lat=tabTemp['lat'].values
lon=tabTemp['lon'].values
lc=tabTemp['LC4'].values.astype(float)+tabTemp['LC5'].values.astype(float)
figM = plt.figure(figsize=(8, 4))
gsM = gridspec.GridSpec(1, 1)   
axM = mapplot.mapPoint(figM, gsM[0, 0],lat,lon,lc, s=16, cb=True,vRange=[0,1])    
figM.show() 

# species + land cover
tabCrdS['Fuel']='NA'
for ind in tabCrdS.index:    
    siteName=tabCrdS.loc[ind]['Site']
    spec=tabAll[tabAll['Site']==siteName]['Fuel'].values[0]
    tabCrdS.at[ind,'Fuel']=spec

tabCrdS1=tabCrdS[tabCrdS['Site'].isin(siteNameLst1)]
tabCrdS2=tabCrdS[~tabCrdS['Site'].isin(siteNameLst1)]
tabCrdS3=tabCrdS2[tabCrdS['count']>60]

a1=tabCrdS1['Fuel'].unique().tolist()
a2=tabCrdS3['Fuel'].unique().tolist()

set(a2)-set(a1)
set(a1)-set(a2)

sLst=list(set(a1).intersection(set(a2)))

tab=tabCrdS3

colLC=['LC{}'.format(x) for x in range(1,10)]
cols=colLC+['Mix','NA']
df=pd.DataFrame(index=sLst,columns=cols,data=0)
for s in sLst:
    v=tab[tab['Fuel']==s][colLC].values
    n=len(np.where(v.sum(axis=1)==0)[0])
    v1=(v>0.8).sum(axis=0)
    v2=v.shape[0]-n-v1.sum()
    v1=np.append(v1,v2)
    v1=np.append(v1,n)
    df.loc[s]=v1

