from pyesgf.search import SearchConnection
nodeUrl = 'https://esgf-node.llnl.gov/esg-search'
conn = SearchConnection(nodeUrl, distrib=False)
ctx = conn.new_context()
dAll = ctx.get_facet_options()
dictSearch = dict(project='CMIP6', activity_id='HighResMIP',
                  nominal_resolution='25 km', frequency='day',
                  variant_label='r1i1p1f1', grid_label='gn',
                  variable='tas,pr', latest=True)
ctx = conn.new_context(**dictSearch)
ctx.hit_count
dd = ctx.get_facet_options()
dd.keys()
dd['activity_id']
dd['experiment_id']
dd['source_id']

ctx1 = ctx.constrain(experiment_id='highresSST-present')
ctx1.facet_counts['source_id']

{'MRI-AGCM3-2-S': 4, 'MRI-AGCM3-2-H': 4, 'IPSL-CM6A-ATM-ICO-VHR': 4,
    'HiRAM-SIT-LR': 3, 'HiRAM-SIT-HR': 4, 'CMCC-CM2-VHR4': 2}

temp = ctx.constrain(
    experiment_id='highresSST-present', source_id='HiRAM-SIT-LR', 
    variable_id='tas')
ds = temp.search()

for kk in ds[0].__dict__['json'].keys():
    print(kk)
    for dd in ds:
        print(dd.__dict__['json'][kk])

for kk in ['replica', 'model_cohort']:
    print(kk)
    for dd in ds:
        print(dd.__dict__['json'][kk])

set(ds[0].__dict__['json'])-set(ds[1].__dict__['json'])


r1 = ctx.search(query='source_id:{}'.format('MRI-AGCM3-2-S'))

for i, rr in enumerate(r1):
    files = rr.file_context().search()
    for j, ff in enumerate(files):
        print(i, j, ff.download_url)


ctx.hit_count
ctx.facet_counts['variant_label']

ctx.facet_counts['variable']

ctx.facet_counts['source_id']

r1 = ctx.search(
    query='experiment_id:{}&variable:pr'.format('highresSST-present'))
r2 = ctx.search(query='experiment_id:{}'.format('highresSST-future'))

files = r2[0].file_context().search()

r2[0].download_url()


qq = experiment_id: {}


r1 = ctx.search(query='experiment_id=highresSST-present&variable=pr,tas')

r2 = ctx.search(query='variable=pr,tas')

r1[0].aggregation_context().__dict__
r1[0].file_context().__dict__


for rr in r1:
    rr.__dict__['json']['experiment_id']

for rr in r2:
    rr.__dict__['json']['experiment_id']
