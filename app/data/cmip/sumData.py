from pyesgf.search import SearchConnection
import pandas as pd
import numpy as np
nodeUrl = 'https://esgf-node.llnl.gov/esg-search'
conn = SearchConnection(nodeUrl, distrib=False)
ctx = conn.new_context()
dictSearch = dict(project='CMIP6', activity_id='HighResMIP',
                  frequency='day', latest=True)

# find out potential models
ctx = conn.new_context(**dictSearch)
dd = ctx.get_facet_options()
# dd.keys()

# seems that only four models
mLst = ['MPI-ESM1-2-XR', 'MPI-ESM1-2-HR',
        'HadGEM3-GC31-MM', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-HM',
        'FGOALS-f3-H',
        'EC-Earth3P-HR', 'EC-Earth3P',
        'CNRM-CM6-1-HR',
        'CMCC-CM2-VHR4', 'CMCC-CM2-HR4']

# find if variable exist, resolution, variant id
varLst = ['pr', 'prc', 'ta', 'tas', 'tasmax', 'tasmin']
colLst = varLst+['res1', 'res2', 'var1', 'var2']
dfVar = pd.DataFrame(columns=colLst, index=mLst)
for mm in mLst:
    print(mm)
    ctx1 = ctx.constrain(experiment_id='hist-1950', source_id=mm)
    ctx2 = ctx.constrain(experiment_id='highres-future', source_id=mm)
    b1 = np.array([var in ctx1.facet_counts['variable'].keys()
                  for var in varLst])
    b2 = np.array([var in ctx2.facet_counts['variable'].keys()
                  for var in varLst])
    dfVar.loc[mm][varLst] = b1+b2*1
    dfVar.loc[mm]['res1'] = list(
        ctx1.facet_counts['nominal_resolution'].keys())
    dfVar.loc[mm]['res2'] = list(
        ctx2.facet_counts['nominal_resolution'].keys())
    dfVar.loc[mm]['var1'] = list(ctx1.facet_counts['variant_label'].keys())
    dfVar.loc[mm]['var2'] = list(ctx1.facet_counts['variant_label'].keys())

# remove CMCC for no temperature
# remvoe HadGEM3-LL for coarse resolution
# count for number of dataset found, until reach 1
mLst = ['MPI-ESM1-2-XR', 'MPI-ESM1-2-HR',
        'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM',
        'FGOALS-f3-H',
        'EC-Earth3P-HR', 'EC-Earth3P',
        'CNRM-CM6-1-HR']

varLst = ['pr', 'prc', 'ta', 'tas', 'tasmax', 'tasmin']
dfVar2 = pd.DataFrame(columns=varLst)
for mm in mLst:
    print(mm)
    ctx1 = ctx.constrain(experiment_id='hist-1950', source_id=mm)
    ctx2 = ctx.constrain(experiment_id='highres-future', source_id=mm)
    for a0, lab in zip([ctx1, ctx2], ['H', 'F']):
        for res in ctx1.facet_counts['nominal_resolution'].keys():
            a1 = a0.constrain(nominal_resolution=res)
            for varLab in a1.facet_counts['variant_label'].keys():
                a2 = a1.constrain(variant_label=varLab)
                mStr = '{},{},{},{}'.format(mm, res, varLab, lab)
                cLst = list()
                for var in varLst:
                    if var in a2.facet_counts['variable'].keys():
                        cLst.append(a2.facet_counts['variable'][var])
                    else:
                        cLst.append(0)
                dfVar2.loc[mStr] = cLst
                print(mStr, cLst)
dfVar2

# check duplicate
temp= ctx.constrain(
    experiment_id='highres-future', source_id='CNRM-CM6-1-HR',
    variant_label='r2i1p1f2',variable='ta')
temp.facet_counts['table_id']

temp= ctx.constrain(
    experiment_id='highres-future', source_id='FGOALS-f3-H',
    variant_label='r1i1p1f1',variable='pr')
temp.facet_counts



for i, rr in enumerate(temp.search()):
    files = rr.file_context().search()
    for j, ff in enumerate(files):
        print(i, j, ff.download_url)

files = r[0].file_context().search()

r[0].file_context().__dict__
r[1].file_context().__dict__
