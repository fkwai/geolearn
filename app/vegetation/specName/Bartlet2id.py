import os
import pandas as pd

# clean up LFMC species to match other database

DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
filePV = os.path.join(DIR_VEG, 'PV-Bartlet2.csv')
tabPV = pd.read_csv(filePV)
specLst = tabPV['Species'].unique().tolist()

# map TRY to Bartlet
fileTrySpec = os.path.join(DIR_VEG, 'try-species.txt')
tabTry = pd.read_table(fileTrySpec, header=0).drop(0)
dictCol = {'try_id': int, 'try_spec': str}
df = pd.DataFrame(index=specLst, columns=list(dictCol.keys()))
for spec in df.index:
    tab = tabTry[tabTry['AccSpeciesName'] == spec]
    if len(tab) == 1:
        df.at[spec, 'try_id'] = tab['AccSpeciesID'].values[0]
        df.at[spec, 'try_spec'] = tab['AccSpeciesName'].values[0]
    elif len(tab) > 1:
        print(tab)
    elif len(tab) ==0:
        print(spec)
