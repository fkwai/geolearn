import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json

outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabAll = pd.read_csv(outFile)
tabAll['Date'] = pd.to_datetime(tabAll['Date'], format='%Y-%m-%d')
tabAll.drop(columns=['siteId_x'], inplace=True)
tabAll.drop(columns=['siteId_y'], inplace=True)

crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)

# add site id
idLst = ['N{:04d}'.format(x + 1) for x in range(len(tabCrd))]
tabCrd['siteId'] = idLst

# assign site id to tabAll by site, gacc, state
matchField=['Site', 'State', 'GACC']
tabOut=pd.merge(tabAll, tabCrd[matchField+['siteId']], on=matchField)

outFile=os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabOut.to_csv(outFile, index=False)
crdFile=os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd=tabCrd.set_index('siteId')
tabCrd.to_csv(crdFile,index='siteId')
