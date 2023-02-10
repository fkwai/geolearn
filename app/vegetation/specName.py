import os
import pandas as pd
import json

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
n0, n1, n2 = [list() for x in range(3)]
countMulti = 0
dictNameLFMC = dict()
for k, spec in enumerate(specLFMC):
    if not pd.isnull(spec) and not spec[:7] == 'Unknown':
        if ',' in spec:
            tempLst = spec.split(',')
            for temp in tempLst:
                if temp[0] == ' ':
                    temp = temp[1:]
            countMulti = countMulti + 1
        else:
            sLst = [spec]
        for kk, ss in enumerate(sLst):
            temp = ss.split(' ')
            if len(temp) == 1:
                temp.append('sp.')
            if temp[1] == 'spp.':
                temp[1] = 'sp.'
            n0.append(temp[0] + ' ' + temp[1])
            if kk == 0:
                dictNameLFMC[spec] = temp[0] + ' ' + temp[1]


for k, spec in enumerate(specPV1):
    if not pd.isnull(spec):
        if '.' in spec:
            temp = spec.split('.')
        else:
            temp = spec.split(' ')
        n1.append(temp[0] + ' ' + temp[1])

for k, spec in enumerate(specPV2):
    temp = spec.split(' ')
    n2.append(temp[0] + ' ' + temp[1])

n0 = set(n0)
n1 = set(n1)
n2 = set(n2)

c1 = sorted(n0.intersection(n1))
c2 = sorted(n0.intersection(n2))
n1.intersection(n2)

# create a name dict
g0, s0, g1, s1, g2, s2 = [list() for x in range(6)]
for n, g, s in zip([n0, n1, n2], [g0, g1, g2], [s0, s1, s2]):
    for nn in n:
        gg, ss = nn.split(' ')
        g.append(gg)
        s.append(ss)

gA = sorted(set(g0 + g1 + g2))
dictName = dict()
for g in gA:
    k0 = [k for k, gg in enumerate(g0) if gg == g]
    k1 = [k for k, gg in enumerate(g1) if gg == g]
    k2 = [k for k, gg in enumerate(g2) if gg == g]
    dictName[g] = {
        'LFMA': [s0[k] for k in k0],
        'Anderegg': [s1[k] for k in k1],
        'Bartlet': [s2[k] for k in k2],
    }
saveName = os.path.join(DIR_VEG, 'speciesName')
with open(saveName + '.json', 'w') as fp:
    json.dump(dictName, fp, indent=4)


for c in c1:
    temp = tabLFMC[tabLFMC['Species collected'] == c]
    temp['Sitename'].unique()

indLst = ['all', 'c1', 'c2']
colLst = ['all', 't1', 't2']
t1 = 20141003
t2 = 20150101
tab = tabLFMC.replace({'Species collected': dictNameLFMC})
countSite = pd.DataFrame(index=indLst, columns=colLst)
countSample = pd.DataFrame(index=indLst, columns=colLst)


for x in indLst:
    for y in colLst:
        tt = tab
        if not x == 'all':
            c = globals()[x]
            tt = tt[tt['Species collected'].isin(c)]
        if not y == 'all':
            t = globals()[y]
            tt = tt[tt['Sampling date'] > t]
        countSample.at[x, y] = len(tt)
        countSite.at[x, y] = len(tt['Sitename'].unique())

ref='National Fuel Moisture Database http://www.wfas.net/nfmd/public/'
tab3=tab[tab['Reference'] == ref]
len(tab3['Sitename'].unique())

tab3=tab2[tab2['Reference'] == ref]
len(tab3['Sitename'].unique())


tab=tabLFMC[tabLFMC['Reference'] == ref]
len(tabLFMC['Sitename'].unique())
len(tab['Sitename'].unique())

fileLFMC2=os.path.join(DIR_VEG, 'NFMD', 'LFMC-NFMD.csv')
tabLFMC2=pd.read_csv(fileLFMC2)
len(tabLFMC2['site'].unique())
