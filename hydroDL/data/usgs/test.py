import hydroDL.data.usgs.read as read
import os
from hydroDL import kPath
import pandas as pd
siteNo='01095220'

df1,df2=read.readSampleRaw(siteNo)

fileC = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
dfC = read.readUsgsText(fileC, dataType='sample')

fileName=fileC
dataType='sample'
with open(fileName) as f:
    k = 0
    line = f.readline()
    while line[0] == "#":
        line = f.readline()
        k = k + 1
    headLst = line[:-1].split('\t')
    typeLst = f.readline()[:-1].split('\t')
pdf = pd.read_table(fileName, header=k, dtype=str)
