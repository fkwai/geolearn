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



    if len(tab) > 1:
        print(tab)
    elif len(tab) == 1:
        df.at[spec, 'try_id'] = tab['AccSpeciesID'].values[0]
        df.at[spec, 'try_spec'] = tab['AccSpeciesName'].values[0]

a=tabTry['AccSpeciesName'].tolist()
spec


SequenceMatcher(None,a[0],spec).ratio()
s.ratio()

outFile = os.path.join(DIR_VEG, 'spec-table-Barlet')
df.to_csv(outFile)

# fixed table
dictCol = {'try_id': int, 'try_spec': str}
fixFile = os.path.join(DIR_VEG, 'spec-table-fix')
dfFix=pd.read_csv(fixFile,index_col=0)
dfFix['try_id'].fillna(0,inplace=True)
dfFix['try_spec'].fillna('NA',inplace=True)
dfFix=dfFix.astype(dtype=dictCol)
idLst=dfFix['try_id'].unique().tolist()
dfFix.compare(df,align_axis=1)

specLst1=['Quercus rubra', 'Tsuga heterophylla', 'Vaccinium parvifolium']
specLst2=['Abies balsamea', 'Abies grandis', 'Alnus rubra',
 'Crataegus monogyna', 'Encelia farinosa', 'Erica arborea', 
 'Gutierrezia sarothrae', 'Juniperus communis', 'Juniperus oxycedrus',
  'Juniperus phoenicea', 'Liquidambar styraciflua', 'Olea europaea', 
  'Picea rubens', 'Pinus ponderosa', 'Prosopis glandulosa', 
  'Quercus rubra', 'Thuja plicata']

dfFix['Barlet']=False
dfFix['Anderegg']=False
for spec in specLst1:
    dfFix.at[spec,'Anderegg']=True    
for spec in specLst2:
    dfFix.at[spec,'Barlet']=True
outFile = os.path.join(DIR_VEG, 'spec-fix')
dfFix.to_csv(outFile)

    
# manual 
pd.concat([df,dfFix]).drop_duplicates(keep=False)

# print for try
sOut=''
for id in idLst[:80]:
    sOut=sOut+str(id)+','
sOut
sOut=''
for id in idLst[80:160]:
    sOut=sOut+str(id)+','
sOut
sOut=''
for id in idLst[160:240]:
    sOut=sOut+str(id)+','
sOut
sOut=''
for id in idLst[240:]:
    sOut=sOut+str(id)+','
sOut

a=[str(id)+',' for id in idLst]