from pyesgf.search import SearchConnection


nodeUrl = 'https://esgf-node.llnl.gov/esg-search'
conn = SearchConnection(nodeUrl, distrib=False)
ctx = conn.new_context()
dictSearch = dict(project='CMIP6', activity_id='HighResMIP',
                  nominal_resolution='50 km', frequency='day',
                  variant_label='r1i1p1f1', latest=True)

dictSearch = dict(project='CMIP6', activity_id='HighResMIP',
                  frequency='day', latest=True)

# find out potential models
ctx = conn.new_context(**dictSearch)
ctx1 = ctx.constrain(experiment_id='hist-1950')
ctx2 = ctx.constrain(experiment_id='highres-future')
ctx1.facet_counts['source_id']
ctx2.facet_counts['source_id']


ctx.constrain(experiment_id='highresSST-present',
              variable='pr').facet_counts['source_id']
ctx.constrain(experiment_id='control-1950',
              variable='pr').facet_counts['source_id']
ctx.constrain(experiment_id='hist-1950',
              variable='pr').facet_counts['source_id']
ctx.constrain(experiment_id='highres-future',
              variable='pr').facet_counts['source_id']
ctx.constrain(experiment_id='highresSST-future',
              variable='pr').facet_counts['source_id']

aa=ctx.constrain(experiment_id='hist-1950',
              variable='pr').facet_counts['source_id']
bb=ctx.constrain(experiment_id='highres-future',
              variable='pr').facet_counts['source_id']

aa.keys()
bb.keys()

ctx.constrain(experiment_id='highresSST-present').facet_counts['source_id']
ctx.constrain(experiment_id='control-1950').facet_counts['source_id']
ctx.constrain(experiment_id='hist-1950').facet_counts['source_id']
ctx.constrain(experiment_id='highres-future').facet_counts['source_id']
ctx.constrain(experiment_id='highresSST-future').facet_counts['source_id']

ctx.constrain(experiment_id='highres-future').facet_counts['short_description']

dd = ctx.get_facet_options()

dd.keys()

# seems that only four models
# NICAM16-9S is not gn and stopped at 1960
# CMCC-CM2-VHR4 only provide pr in future
mLst = ['MRI-AGCM3-2-S', 'MRI-AGCM3-2-H', 'HiRAM-SIT-HR', 'CMCC-CM2-VHR4']

# find common available variables

varLst = list()
varNameLst = list()
for mm in mLst:
    ctx1 = ctx.constrain(experiment_id='highresSST-present', source_id=mm)
    ctx2 = ctx.constrain(experiment_id='highresSST-future', source_id=mm)
    print(mm)
    print(ctx1.facet_counts['variable'].keys())
    print(ctx2.facet_counts['variable'].keys())
    varLst.append(list(ctx1.facet_counts['variable'].keys()))
    varLst.append(list(ctx2.facet_counts['variable'].keys()))
    varNameLst.append(list(ctx1.facet_counts['variable_long_name'].keys()))
    varNameLst.append(list(ctx2.facet_counts['variable_long_name'].keys()))


varCom = list(set.intersection(*map(set, varLst)))
varNameCom = list(set.intersection(*map(set, varNameLst)))


ctx1 = ctx.constrain(source_id='CMCC-CM2-VHR4')
ctx1.facet_counts['variable_long_name']
ctx1.facet_counts['experiment_id']
ctx1.facet_counts['experiment_title']
