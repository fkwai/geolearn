import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from hydroDL import kPath

tabSite = pd.read_csv(os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv'))
tabSite['slope'] = ''
tabSite['elevation'] = ''
tabSite['aspect'] = ''

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
        if cols[0].text == 'Slope':
            tabSite['slope'] = cols[1].text

# tabSite=tabSite.drop('index',axis=1)
# tabSite.to_csv(os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv'))

siteFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabSite = pd.read_csv(siteFile, index_col='siteId')

siteId = 'N0001'
row = tabSite.loc[siteId]
url = (
    'https://www.wfas.net/nfmd/include/site_page.php?site={}&gacc={}&state={}&grup={}'
).format(row['Site'], row['GACC'], row['State'], row['Group'])
r = requests.get(url)

soup = BeautifulSoup(r.text, 'lxml')
table = soup.find('table')
rows = soup.find_all('tr')
