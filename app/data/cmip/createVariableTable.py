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


dictVar = dict()
for mm in mLst:
    print(mm)
    for expId in ['hist-1950', 'highres-future']:
        temp = ctx.constrain(experiment_id=expId, source_id=mm)
        for var in temp.facet_counts['variable'].keys():
            tempVar = temp.constrain(variable=var)
            longName = list(
                tempVar.facet_counts['variable_long_name'].keys())[0]
            if var in dictVar:
                longNameTemp = dictVar[var]
                if longNameTemp != longName:
                    print(mm, expId, var)
                    print(longName)
                    print(longNameTemp)
            else:
                dictVar[var] = longName
for key in sorted(dictVar):
    print("{}='{}',".format(key, dictVar[key]))
