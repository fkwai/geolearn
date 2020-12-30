from hydroDL.data import gageII
import numpy as np
import pandas as pd
import os

varLst = ['ECO2_BAS_DOM', 'ECO3_BAS_DOM']
dfR = gageII.readData(varLst=varLst)
dfR = gageII.updateCode(dfR)

fileEco3 = r'C:\Users\geofk\work\map\ecoRegion\tabEco3.csv'
tabEco3 = pd.read_csv(fileEco3)

fileLookup = os.path.join(gageII.dirTab, 'conterm_x_ecoregion3_names.csv')
tabLookup = pd.read_csv(fileLookup)

len(np.sort(dfR['ECO3_BAS_DOM'].unique()))
codeLst = list(range(1, 85))
dfT = pd.DataFrame(index=codeLst, columns=['Eco2', 'Eco3', 'Eco3_Name'])
for code in codeLst:
    eco2 = dfR[dfR['ECO3_BAS_DOM'] == code]['ECO2_BAS_DOM'].unique()
    eco3Name = tabLookup[tabLookup['ECO3_CODE'] == code]['ECO3_NAME'].values
    if len(eco3Name) == 1:
        eco3 = tabEco3[tabEco3['NA_L3NAME'] == eco3Name[0]]['NA_L3CODE'].values
        dfT.at[code, 'Eco3_Name'] = eco3Name[0]
    if len(eco2) == 1:
        dfT.at[code, 'Eco2'] = eco2[0]
    if len(eco3) == 1:
        dfT.at[code, 'Eco3'] = eco3[0]

fileT = os.path.join(gageII.dirTab, 'EcoTab.csv')
dfT.to_csv(fileT)

# then manual work
fileT = os.path.join(gageII.dirTab, 'lookupEco.csv')
