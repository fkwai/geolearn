import os
import pandas as pd
from hydroDL import kPath


DIR_RAW = os.path.join(kPath.dirVeg, 'NFMD', 'raw')

tabLst = list()
for file in os.listdir(DIR_RAW):
    if file.endswith('.txt'):
        df = pd.read_table(os.path.join(DIR_RAW, file))
        tabLst.append(df)
tab = pd.concat(tabLst, ignore_index=True)

# clean up
tab = tab.drop('Unnamed: 7', axis=1)
tab['Date'] = pd.to_datetime(tab['Date'])
# tab = tab[tab['Date'] >= pd.to_datetime('2015-01-01')]
tab = tab[tab['Percent'] <= 1000]
rmFuel = [
    '1-Hour',
    '10-Hour',
    '100-Hour',
    '1000-Hour',
    '1-hour',
    '10-hour',
    '100-hour',
    '1000-hour',
    'Duff (DC)',
    'Moss, Dead (DMC)',
]
tab=tab[tab['Fuel'].isin(rmFuel)==False]
outFile=os.path.join(kPath.dirVeg, 'NFMD', 'NFMD-20230315.csv')
tab.to_csv(outFile, index=False)