import os
import pandas as pd

DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
fileLFMC = os.path.join(DIR_VEG, 'LFMC-global.csv')
filePV1 = os.path.join(DIR_VEG, 'PV-Anderegg.csv')
filePV2 = os.path.join(DIR_VEG, 'PV-Bartlet.csv')

tabLFMC = pd.read_csv(fileLFMC)
tabPV1 = pd.read_csv(filePV1)
tabPV2 = pd.read_csv(filePV2)

specLFMC = tabLFMC['Species collected'].unique().tolist()
specPV1 = tabPV1['Species'].unique().tolist()
specPV2 = tabPV2['Species'].unique().tolist()

# cleanup LFMC spec
gL, sL, gP1, sP1, gP2, sP2 = [list() for x in range(6)]

for k, spec in enumerate(specLFMC):
    if not pd.isnull(spec) and not spec[:7] == 'Unknown':
        if ',' in spec:
            tempLst = spec.split(',')
            for temp in tempLst:
                if temp[0] == ' ':
                    temp = temp[1:]
        else:
            sLst = [spec]
        for ss in sLst:
            temp = ss.split(' ')
            gL.append(temp[0])
            if len(temp) == 1:
                temp.append('sp.')
            if temp[1] == 'spp.':
                temp[1] = 'sp.'
            sL.append(temp[1])
            if temp[0] == '':
                print(k)

for k, spec in enumerate(specPV1):
    if not pd.isnull(spec):
        if '.' in spec:
            temp = spec.split('.')
        else:
            temp = spec.split(' ')
        gP1.append(temp[0])
        sP1.append(temp[1])

for k, spec in enumerate(specPV2):
    temp = spec.split(' ')
    gP2.append(temp[0])
    sP2.append(temp[1])


s0 = [g+' '+s for g, s in zip(gL, sL)]
s1 = [g+' '+s for g, s in zip(gP1, sP1)]
s2 = [g+' '+s for g, s in zip(gP2, sP2)]
c1 = sorted(set(s0).intersection(set(s1)))
c2 = sorted(set(s0).intersection(set(s2)))
sorted(set(s1).intersection(set(s2)))

# create a name dict
gA = sorted(set(gL+gP1+gP2))

dictName = dict()
g = gA[0]
k0 = [k for k, gg in enumerate(gL) if gg == g]
k1 = [k for k, gg in enumerate(gP1) if gg == g]
k2 = [k for k, gg in enumerate(gP2) if gg == g]
dictName[g] = {'LFMA': [sL[k] for k in k0],
               'Anderegg': [sP1[k] for k in k1],
               'Bartlet': [sP2[k] for k in k2]}

for c in c1:
    temp = tabLFMC[tabLFMC['Species collected'] == c]
    temp['Sitename'].unique()

sc = 0
oc = 0
for c in c2:
    c = c2[14]
    temp = tabLFMC[tabLFMC['Species collected'] == c]
    sc = sc+len(temp['Sitename'].unique())
    oc = oc+len(temp)

c = c2[13]
temp = tabLFMC[tabLFMC['Species collected'] == c]
len(temp['Sitename'].unique())
len(temp)
c = c2[14]
temp = tabLFMC[tabLFMC['Species collected'] == c]
len(temp['Sitename'].unique())
len(temp)
