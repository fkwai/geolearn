import hydroDL.data.usgs.read as read
import os
import hydroDL.kPath as kPath
import pandas as pd

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileG = os.path.join(dirInv, 'inv-gageII')
fileQ = os.path.join(dirInv, 'inv-surfaceWater')
fileC = os.path.join(dirInv, 'inv-waterQuality')

tabG = read.readUsgsText(fileG)
tabQ = read.readUsgsText(fileQ)
tabC = read.readUsgsText(fileC)
sG = tabG['site_no'].tolist()
sQ = tabQ['site_no'].tolist()
sC = tabC['site_no'].tolist()

# save to csv
import csv

tabG.to_csv(fileG + '.csv', quoting=csv.QUOTE_NONNUMERIC)
tabQ.to_csv(fileQ + '.csv', quoting=csv.QUOTE_NONNUMERIC)
tabC.to_csv(fileC + '.csv', quoting=csv.QUOTE_NONNUMERIC)

# find intersections between
len(sQ)
len(sC)
len(set(sG) - set(sQ))
len(set(sG) - set(sC))
len(set(sC) - set(sQ) - set(sG))
len(set(sQ) - set(sC) - set(sG))
len(set(sG) - set(sC) - set(sQ))
len(set(sG).intersection(set(sC)).intersection(set(sQ)))
len(set(sQ).intersection(set(sC)))
len(set(sQ).intersection(set(sC)) - set(sG))
len(set(sG).intersection(set(sC)) - set(sQ))
len(set(sG).intersection(set(sQ)) - set(sC))

# investigate streamflow sites
dirQ = os.path.join(kPath.dirUsgs, 'streamflow', 'csv')
s0 = os.listdir(dirQ)
d1 = sorted(set(sG) - set(s0))
d2 = sorted(set(sG) - set(sQ))
len(set(d1).intersection(set(d2)))
# 112 stations do not have daily statistic

# investigate water quality sites
dirC = os.path.join(kPath.dirUsgs, 'sample', 'csvAll')
s0 = list()
for s in os.listdir(dirC):
    if not s[-5:] == '_flag':
        s0.append(s)
s1 = tabG[tabG['qw_count_nu'] > 0]['site_no'].tolist()
s2 = set(sG).intersection(set(sC))

set(s1) - set(s0)
set(s2) - set(s1)

d1 = sorted(set(sG) - set(s0))
d2 = sorted(set(sG) - set(s2))
set(d1) - set(d2)
set(d2) - set(d1)


len(set(d1).intersection(set(d2)))


''' extract a gageII list to extract inventory
siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
siteLst=dfSite['siteNo'].tolist()
with open('temp', 'w') as fp:
    for s in siteLst:
        # write each item on a new line
        fp.write('%s\n' % s)
    print('Done')
'''

# there are 5 duplicated site-id
sd = tabG[tabG.duplicated('site_no')]['site_no'].tolist()

for s in sd:
    tab[tab['site_no'] == s]

set(s0) - set(tab['site_no'])

set(tab['site_no']) - set(s0)
tab.to_csv('inventory-gageII')

tab.columns

sum(tab['qw_end_date'])

sum(tab['qw_end_date'] > '1979-01-01')
