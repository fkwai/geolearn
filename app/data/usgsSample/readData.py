# upgrade code to read flags and save CSV
from hydroDL.data import usgs
from hydroDL import kPath
from hydroDL.app import waterQuality
import os
import pandas as pd

pd.set_option('display.max_rows', 100)

siteNo = '07060710'
codeLst = usgs.codeLst
startDate = pd.datetime(1979, 1, 1)

fileC = os.path.join(kPath.dirData, 'USGS', 'sample', siteNo)
dfC = usgs.readUsgsText(fileC, dataType='sample')
if startDate is not None:
    dfC = dfC[dfC['date'] >= startDate]
dfC = dfC.set_index('date')

codeSel = list(set(codeLst) & set(dfC.columns.tolist()))
codeSel_cd = [code + '_cd' for code in codeSel]
dfC = dfC[codeSel+codeSel_cd].dropna(how='all')
dfC1 = dfC[codeSel]
dfC2 = dfC[codeSel_cd]
dfC2[dfC1.notna().values & dfC2.isna().values] = 'x'
dfC2 = dfC2.fillna('')
bDup = dfC.index.duplicated(keep=False)
indUni = dfC.index[~bDup]
indDup = dfC.index[bDup].unique()
indAll = dfC.index.unique()
dfO1 = pd.DataFrame(index=indAll, columns=codeSel)
dfO2 = pd.DataFrame(index=indAll, columns=codeSel_cd)
dfO1.loc[indUni] = dfC1.loc[indUni][codeSel]
dfO2.loc[indUni] = dfC2.loc[indUni][codeSel_cd]
for ind in indDup:
    temp1 = dfC1.loc[ind]
    temp2 = dfC2.loc[ind]
    for code in codeLst:
        if 'x' in temp2[code+'_cd'].tolist():
            dfO1.loc[ind][code] = temp1[code][temp2[code+'_cd'] == 'x'].mean()
            if temp2[code+'_cd'].tolist().count('x') > 1:
                dfO2.loc[ind][code+'_cd'] = 'X'
            else:
                dfO2.loc[ind][code+'_cd'] = 'x'
        else:
            dfO1.loc[ind][code] = temp1[code].mean()
            dfO2.loc[ind][code+'_cd'] = ''.join(temp2[code+'_cd'])


dirC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv')
dfO1.to_csv(os.path.join(dirC, siteNo))
dfO2.to_csv(os.path.join(dirC, siteNo+'_flag'))

siteNo = '01013500'
# siteNo = '07060710'
dirC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv')
fileC1 = os.path.join(dirC, siteNo)
dfO1 = pd.read_csv(fileC1)
dfO1['date'] = pd.to_datetime(dfO1['date'], format='%Y-%m-%d')
dfO1 = dfO1.set_index('date')
fileC2 = os.path.join(dirC, siteNo+'_flag')
dfO2 = pd.read_csv(fileC2)
dfO2['date'] = pd.to_datetime(dfO2['date'], format='%Y-%m-%d')
dfO2 = dfO2.set_index('date')

dfO3 = pd.DataFrame(index=dfO2.index, columns=dfO2.columns)
dfO3[(dfO2 == 'x') | (dfO2 == 'X')] = 0
dfO3[(dfO2 != 'x') & (dfO2 != 'X') & (dfO2.notna())] = 1


dfO2.where(dfO2 != 'x', other=0)

dfO2.notna()
