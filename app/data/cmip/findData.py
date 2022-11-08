from pyesgf.search import SearchConnection


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

# find common available variables
varLst = list()
varNameLst = list()
for mm in mLst:
    ctx1 = ctx.constrain(experiment_id='hist-1950', source_id=mm)
    ctx2 = ctx.constrain(experiment_id='highres-future', source_id=mm)
    print(mm)
    print(ctx1.facet_counts['variable'].keys())
    print(ctx2.facet_counts['variable'].keys())
    varLst.append(list(ctx1.facet_counts['variable'].keys()))
    varLst.append(list(ctx2.facet_counts['variable'].keys()))
    varNameLst.append(list(ctx1.facet_counts['variable_long_name'].keys()))
    varNameLst.append(list(ctx2.facet_counts['variable_long_name'].keys()))
varCom = list(set.intersection(*map(set, varLst)))
varNameCom = list(set.intersection(*map(set, varNameLst)))


ctx1 = ctx.constrain(experiment_id='hist-1950', source_id='CMCC-CM2-VHR4')
ctx1.facet_counts['variable']
ctx1 = ctx.constrain(experiment_id='hist-1950', source_id='CMCC-CM2-VHR4',
                     variable='pr')
ctx1.facet_counts['variable']
ctx1.facet_counts['variable_long_name']



ctx1 = ctx.constrain(experiment_id='hist-1950', source_id='CMCC-CM2-VHR4',
                     cf_standard_name='wind_speed')
ctx1.get_facet_options()

ctx1 = ctx.constrain(experiment_id='hist-1950', source_id='CMCC-CM2-VHR4',
                     cf_standard_name='air_temperature')
ctx1.get_facet_options()

ctx1.facet_counts['variable_long_name']
ctx1.facet_counts['experiment_id']
ctx1.facet_counts['experiment_title']


ctx2.facet_counts['variable_id']
ctx2.facet_counts['table_id']
