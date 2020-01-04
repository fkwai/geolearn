from hydroDL.data import usgs
import pandas as pd

# read site inventory
fileName = r'C:\Users\geofk\work\waterQuality\tsDaily\all-streamflow-site.txt'
siteAll = usgs.readUsgsText(fileName)

indLst = list()
for ind, row in siteAll.iterrows():
    strT = row['data_types_cd']
    if strT[0] != 'N' and strT[2] != 'N':
        indLst.append(ind)
siteWq = siteAll.iloc[indLst, :]
tabM=siteWq['instruments_cd'].value_counts()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(tabM)

# not right...
siteAll.query('instruments_cd == "NNNYNNNNNNNYNNNNYNNNNNNNNNNNNN"')
