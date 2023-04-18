import os
import requests
import pdb
from bs4 import BeautifulSoup
import pandas as pd
from hydroDL import kPath

outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tab = pd.read_csv(outFile)
tabSite = tab[['GACC', 'State', 'Group', 'Site']].drop_duplicates().reset_index()
tabSite['lat'] = ''
tabSite['lon'] = ''

for k, row in tabSite.iterrows():
    print(k, row['Site'], row['GACC'], row['State'], row['Group'])
    if row['lat'] != '':
        continue
    url = (
        'https://www.wfas.net/nfmd/include/site_page.php?site={}&gacc={}&state={}&grup={}'
    ).format(row['Site'], row['GACC'], row['State'], row['Group'])
    r = requests.get(url)
    if r.status_code != 200:
        print('site not found')
        continue
    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find('table')
    rows = soup.find_all('tr')
    for tr in rows:
        cols = tr.find_all('td')
        if cols[0].text == 'Location':
            crdStr = cols[1].text.split('x')
            latStr = crdStr[0][:-1].split('-')
            lonStr = crdStr[1][1:].split('-')
            if len(latStr) == 3 and len(lonStr) == 3:
                lat = float(latStr[0]) + float(latStr[1]) / 60 + float(latStr[2]) / 3600
                lon = float(lonStr[0]) + float(lonStr[1]) / 60 + float(lonStr[2]) / 3600
                tabSite.loc[k, 'lat'] = lat
                tabSite.loc[k, 'lon'] = -lon
            else:
                print('ERROR', crdStr)

tabSite=tabSite.drop('index',axis=1)
tabSite.to_csv(os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv'))