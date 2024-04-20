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

# df1.columns.tolist()
aa=df1['date'].unique()
aa.sort()



fileNFMD='/home/kuai/work/VegetationWater/data/NFMD/NFMD.csv'
df2=pd.read_csv(fileNFMD)
df2.drop(df2[df2['GACC']=='AICC'].index)
df2.drop(df2[df2['GACC']=='EACC'].index)

siteNameLst2=df2['Site'].unique().tolist()
len(list(set(siteNameLst1).intersection(siteNameLst2)))
set(siteNameLst1)-set(siteNameLst2)
sd=np.datetime64('2015-01-01')
ed=np.datetime64('2019-03-01')
df2['Date']=pd.to_datetime(df2['Date'])
df3=df2[(df2['Date']>=sd) & (df2['Date']<=ed)]
siteNameLst3=df3['Site'].unique().tolist()

siteLstPlot=list(set(siteNameLst3)-set(siteNameLst1))
fileSite=r'/home/kuai/work/VegetationWater/data/NFMD/NFMDsite.csv'
dfSite=pd.read_csv(fileSite)

sitePlot,lat,lon, count,countMin=list(),list(),list(),list(),list()
for site in siteLstPlot:
# for site in siteNameLst1:
    sitePlot.append(site)
    lat.append(dfSite['lat'][dfSite['Site']==site].values[0])
    lon.append(dfSite['lon'][dfSite['Site']==site].values[0])
    dfTemp=df3[df3['Site']==site]
    sLst=dfTemp['Fuel'].unique().tolist()
    countTemp=[]
    for s in sLst:
        temp=dfTemp[dfTemp['Fuel']==s]
        countTemp.append(len(temp))     
    count.append(np.max(np.array(countTemp)))
    countMin.append(np.min(np.array(countTemp)))
    
lat=np.array(lat)
lon=np.array(lon)
count=np.array(count)
countMin=np.array(countMin)

ind=np.where(count>20)[0]

def funcM():
    figM = plt.figure(figsize=(8, 4))
    gsM = gridspec.GridSpec(1, 1)
    # axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, count, s=16, cb=True,vRange=[0,50])    
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat[ind], lon[ind], count[ind], s=16, cb=True,vRange=[0,50])    
    figP,axP=plt.subplots(1,1)    
    return figM, axM, figP, axP, lon[ind], lat[ind]


def funcP(iP, axP):
    print(iP)
    # siteName = sitePlot[iP]
    siteName = sitePlot[ind[iP]]
    dfTemp=df3[df3['Site']==siteName]
    sLst=dfTemp['Fuel'].unique().tolist()
    for s in sLst:
        temp=dfTemp[dfTemp['Fuel']==s]
        t=temp['Date'].values
        v=temp['Percent'].values
        axP.plot(t,v,'-*',label=s)   
    axP.legend()

figM, figP = figplot.clickMap(funcM, funcP)


###########

fileName=r'/home/kuai/GitHUB/lfmc_from_sar/input_data/lfmc_training_samples_updated_2019-04-29.csv'
dfOld=pd.read_csv(fileName)

# siteNameLst2=dfOld['Site'].unique().tolist()
# len(list(set(siteNameLst1).intersection(siteNameLst2)))
# set(siteNameLst1)-set(siteNameLst2)
sd=np.datetime64('2015-01-01')
ed=np.datetime64('2019-03-01')
dfOld['date']=pd.to_datetime(dfOld['date'])
dfOld2=dfOld[(dfOld['date']>=sd)]
aa=dfOld2['site'].unique().tolist()

bb=dfOld['site'].unique().tolist()


# group?
a1=dfSite['GACC'].unique().tolist()
dfSiteOld=dfSite[dfSite['Site'].isin(siteNameLst1)]
a2=dfSiteOld['GACC'].unique().tolist()

set(a1)-set(a2)

dfTemp=dfSite[dfSite['GACC']=='EACC']
figM = plt.figure(figsize=(8, 4))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0],dfTemp['lat'].values, dfTemp['lon'].values, dfTemp['lon'].values, s=16, cb=True,vRange=[0,50])    
figM.show()