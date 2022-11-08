from pyesgf.search import SearchConnection
import pandas as pd
import numpy as np

# write urls to a file

nodeUrl = 'https://esgf-node.llnl.gov/esg-search'
conn = SearchConnection(nodeUrl, distrib=False)
ctx = conn.new_context()
dictSearch = dict(project='CMIP6', activity_id='HighResMIP',
                  frequency='day', experiment_id='hist-1950,highres-future',
                  latest=True)

dictLst = [
    dict(source_id='CNRM-CM6-1-HR', variant_label='r2i1p1f2',
         variable='pr,tas,tasmin,tasmax', nominal_resolution='50 km'),
    dict(source_id='EC-Earth3P', variant_label='r3i1p2f1',
         variable='pr,tas,tasmin,tasmax', nominal_resolution='50 km'),
    dict(source_id='EC-Earth3P-HR', variant_label='r1i1p2f1',
         variable='pr,tas,tasmin,tasmax', nominal_resolution='50 km'),
    dict(source_id='HadGEM3-GC31-HM', variant_label='r1i1p1f1',
         variable='pr,tas,tasmin,tasmax', nominal_resolution='50 km'),
    dict(source_id='MPI-ESM1-2-XR', variant_label='r1i1p1f1',
         variable='pr,tas,tasmin,tasmax', nominal_resolution='50 km'),
    dict(source_id='FGOALS-f3-H', variant_label='r1i1p1f1',
         variable='pr,tas', nominal_resolution='25 km',
         data_node='esgf-data1.llnl.gov', experiment_id='highres-future'),
    dict(source_id='FGOALS-f3-H', variant_label='r1i1p1f1',
         variable='pr,tas', nominal_resolution='25 km',
         experiment_id='hist-1950')
]

# find out data
f = open('downloadCMIP6', 'a')
for dd in dictLst:
    dd['source_id']
    dictTemp = dictSearch.copy()
    dictTemp.update(dd)
    ctx = conn.new_context(**dictTemp)
    if ctx.hit_count != len(dd['variable'].split(','))*2:
        print('something wrong', ctx.hit_count)
        print(dd)
    else:
        for i, rr in enumerate(ctx.search()):
            files = rr.file_context().search()
            for j, ff in enumerate(files):
                f.write(ff.download_url)
f.close()

for i, rr in enumerate(ctx.search()):
    files = rr.file_context().search()
    for j, ff in enumerate(files):
        print(i, j, ff.download_url)
