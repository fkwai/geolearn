
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
import json
from hydroDL.post import axplot, figplot, mapplot
import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json


data_file='/home/kuai/GitHUB/lfmc_from_sar/input_data/lstm_input_data_pure+all_same_28_may_2019_res_SM_gap_3M'
df1= pd.read_pickle(data_file)
siteNameLst1=df1['site'].unique().tolist()
siteNameLst1.remove('Panter')
siteNameLst1.append('Panther')


outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabAll = pd.read_csv(outFile)
tabAll['Date'] = pd.to_datetime(tabAll['Date'], format='%Y-%m-%d')
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)

dictMix = list()
for k, site in enumerate(siteNameLst1):
    print(k, site)
    crd = tabCrd[tabCrd['Site']==site]
    tab = tabAll[tabAll['Site'] == site]    
    # tab = tabAll[tabAll['siteId'] == site]
    tab = tab.sort_values(by=['Date'])
    fuelLst = tab['Fuel'].unique().tolist()
    tLst = list()
    vLst = list()
    
    if len(fuelLst)==1:
        siteDict = dict(
            siteId=site,
            siteName=crd['Site'].values[0],
            state=crd['State'].values[0],
            fuel=tab['Fuel'].iloc[0],
            gacc=crd['GACC'].values[0],
            crd=[crd['lat'].values[0], crd['lon'].values[0]],
            t=[d.strftime('%Y-%m-%d') for d in tab['Date'].tolist()],
            v=tab['Percent'].tolist(),
        )
    else:       
        for fuel in fuelLst:
            tabTemp = tab[tab['Fuel'] == fuel]
            tLst.append(tabTemp['Date'].tolist())
            vLst.append(tabTemp['Percent'].tolist())
        tOut, indLst = utils.intersectMulti(tLst)
        mat = np.zeros([len(fuelLst), len(tOut)])
        for k, ind in enumerate(indLst):
            mat[k, :] = np.array(vLst[k])[ind]
        vOut = np.mean(mat, axis=0)
        
        siteDict = dict(
            siteId=site,
            siteName=crd['Site'].values[0],
            state=crd['State'].values[0],
            fuel=fuelLst,
            gacc=crd['GACC'].values[0],
            crd=[crd['lat'].values[0], crd['lon'].values[0]],
            t=[d.strftime('%Y-%m-%d') for d in tOut],
            v=vOut.tolist(),
        )
        dictMix.append(siteDict)
outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_K150.json')
with open(outFile, 'w') as fp:
    json.dump(dictMix, fp, indent=4)