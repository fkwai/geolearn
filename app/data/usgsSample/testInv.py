import hydroDL.data.usgs.read as read
import os 
import hydroDL.kPath as kPath
import pandas as pd

siteNo='02294655'
# siteNo='02294650'
tab=read.readSampleRaw(siteNo)
fileC = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
dfC = read.readUsgsText(fileC, dataType='sample')

with open(fileC) as f:
    k = 0
    line = f.readline()
    while line[0] == "#":
        line = f.readline()
        k = k + 1
    headLst = line[:-1].split('\t')
    typeLst = f.readline()[:-1].split('\t')
pdf = pd.read_table(fileC, header=k, index_col=None,dtype=str)

file1='/home/kuai/Downloads/daily'
file2='/home/kuai/Downloads/waterQuality'
file3='/home/kuai/Downloads/dailyAndWaterQuality'
file4='/home/kuai/Downloads/siteVisit'

tab1=read.readUsgsText(file1)
tab2=read.readUsgsText(file2)
tab3=read.readUsgsText(file3)
tab4=read.readUsgsText(file4)


siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
s0=dfSite['siteNo'].tolist()
s1=tab1['site_no'].tolist()
s2=tab2['site_no'].tolist()
s3=tab3['site_no'].tolist()
s4=tab4['site_no'].tolist()

len(set(s0)-set(s1))
len(set(s0)-set(s2))
len(set(s0)-set(s3))
len(set(s0)-set(s4))


with open('temp', 'w') as fp:
    for s in s0:
        # write each item on a new line
        fp.write("%s\n" % s)
    print('Done')

fileInv='/home/kuai/Downloads/inv-gageII'
tab=read.readUsgsText(fileInv)

set(s0)-set(tab['site_no'])

set(tab['site_no'])-set(s0)
tab.to_csv('inventory-gageII')

tab.columns

sum(tab['qw_end_date'])

sum(tab['qw_end_date']>'1979-01-01')

