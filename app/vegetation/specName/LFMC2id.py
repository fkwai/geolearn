import os
import pandas as pd
import numpy as np


# clean up LFMC species to match other database
DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
fileLFMC = os.path.join(DIR_VEG, 'LFMC-global')
tabLFMC = pd.read_csv(fileLFMC)
tabLFMC = tabLFMC[tabLFMC['Species collected'].notna()]
specLFMC = tabLFMC['Species collected'].unique().tolist()

# add id
temp = tabLFMC['ID'].str.split('_', 2, expand=True)
siteId = temp[0].str.cat(temp[1], sep='_')
siteIdLst = siteId.unique().tolist()
tabLFMC['siteId'] = siteId

# output species and num of samples
dictCol = {
    'nSite': int,
    'nObs': int,
    'nSite_15': int,
    'nObs_15': int,
    'LFMC_spec_clean': str,
    'try_id': int,
    'try_spec': str,
    'nDMC': int,
    'nDMC_site': int,
    'Bartlet_spec': str,
}
dfOut = pd.DataFrame(index=sorted(specLFMC), columns=list(dictCol.keys()))
outFile = os.path.join(DIR_VEG, 'specMatch', 'summary-LFMC')


# fill in LFMC data
for spec in sorted(specLFMC):
    tab = tabLFMC[tabLFMC['Species collected'] == spec]
    tabA = tab[tab['Sampling date'] > 20150101]
    dfOut.loc[spec][['nSite', 'nObs', 'nSite_15', 'nObs_15']] = [
        len(tab['siteId'].unique()),
        len(tab),
        len(tabA['siteId'].unique()),
        len(tabA),
    ]
dfOut.to_csv(outFile)

# clean species name
for spec in dfOut.index:
    temp = spec.split(' ')
    if 'sp.' in temp:
        temp[temp.index('sp.')] = 'sp'
    if spec[:7] == 'Unknown':
        dfOut.at[spec, 'LFMC_spec_clean'] = 'unknown'
    elif ',' in spec:
        dfOut.at[spec, 'LFMC_spec_clean'] = 'mixed'
    elif len(temp) > 2:
        if temp[2] == 'var.':
            print(temp)
            specTemp = '{} {} var. {}'.format(temp[0], temp[1], temp[3])
        else:
            specTemp = '{} {}'.format(temp[0], temp[1])
        dfOut.at[spec, 'LFMC_spec_clean'] = specTemp.strip()
    elif len(temp) == 2:
        specTemp = '{} {}'.format(temp[0], temp[1])
        dfOut.at[spec, 'LFMC_spec_clean'] = specTemp.strip()
    else:
        dfOut.at[spec, 'LFMC_spec_clean'] = 'others'
dfOut.to_csv(outFile)

# map TRY to LFMC
fileTrySpec = os.path.join(DIR_VEG, 'try-species.txt')
tabTry = pd.read_table(fileTrySpec, header=0).drop(0)
specUnmatch = []
for spec in dfOut.index:
    specClean = dfOut['LFMC_spec_clean'].loc[spec]
    tab = tabTry[tabTry['AccSpeciesName'] == specClean]
    if len(tab) == 1:
        dfOut.at[spec, 'try_id'] = tab['AccSpeciesID'].values[0].astype(int)
        dfOut.at[spec, 'try_spec'] = tab['AccSpeciesName'].values[0]
    elif len(tab) > 1:
        print(tab)
    elif len(tab) == 0:
        # output potential matches
        dfOut.at[spec, 'try_id'] = 0
        if specClean == 'mixed':
            dfOut.at[spec, 'try_spec'] = 'mixed'
        elif specClean == 'unknown':
            dfOut.at[spec, 'try_spec'] = 'unknown'
        else:
            dfOut.at[spec, 'try_spec'] = 'unmatched'
            specUnmatch.append(specClean)
dfOut.to_csv(outFile)

""" find potential match for unmatched species """
# # add genus and species to try species
# temp = tabTry['AccSpeciesName'].str.split(' ', 1, expand=True)
# tabTry['genus'] = temp[0]
# tabTry['species'] = temp[1]
# dictFix = dict()
# # find top 5 potential matches
# from difflib import SequenceMatcher
# specLst1 = specUnmatch
# for j, spec1 in enumerate(specLst1):
#     print(j)
#     # find type match
#     temp = spec1.split(' ')
#     tab = tabTry[tabTry['genus'] == temp[0]]
#     specLst2 = tab['AccSpeciesName'].tolist()
#     if len(specLst2)>0:
#         rLst = []
#         for i, spec2 in enumerate(specLst2):
#             rLst.append(SequenceMatcher(None, spec1.lower(), spec2.lower()).ratio())
#         rAry = np.array(rLst)
#         idx = np.argsort(rAry)[::-1]
#         dictFix[spec1] = [specLst2[idx[x]] for x in range(min(5, len(idx)))]
#     else:
#         dictFix[spec1] = ['no genus match']

# fixFile = os.path.join(DIR_VEG, 'specMatch', 'fix-LFMC')
# with open(fixFile, 'w') as f:
#     for key, val in dictFix.items():
#         f.write('{}\t{}\n'.format(key, val))
fileFix = os.path.join(DIR_VEG, 'specMatch', 'fix-LFMC-manual')
dfFix = pd.read_csv(fileFix, header=0)
for specClean, specTry in zip(dfFix['LFMC_spec_clean'], dfFix['try_spec']):
    print(specClean, specTry)
    specLFMC = dfOut[dfOut['LFMC_spec_clean'] == specClean].index.values[0]
    if not specTry == 'unmatched':
        dfOut.at[specLFMC, 'try_spec'] = specTry
        dfOut.at[specLFMC, 'try_id'] = (
            tabTry[tabTry['AccSpeciesName'] == specTry]['AccSpeciesID']
            .values[0]
            .astype(int)
        )
dfOut.to_csv(outFile)

# write try id to file
tryIdFile = os.path.join(DIR_VEG, 'specMatch', 'try-id')
with open(tryIdFile, 'w') as f:
    tryIdLst = dfOut['try_id'].tolist()
    # remove  duplicate and 0 from list
    tryIdLst = sorted(list(set(tryIdLst)))
    tryIdLst.remove(0)
    for tryId in tryIdLst:
        _ = f.write('{},'.format(tryId))

# write after 15 species
tryIdFile = os.path.join(DIR_VEG, 'specMatch', 'try-id-15')
with open(tryIdFile, 'w') as f:
    tryIdLst=dfOut['try_id'][dfOut['nObs_15']>0].tolist()
    tryIdLst = sorted(list(set(tryIdLst)))
    tryIdLst.remove(0)
    for tryId in tryIdLst:
        _ = f.write('{},'.format(tryId))


