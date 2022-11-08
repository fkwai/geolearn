import os
import urllib
from pyesgf.search import SearchConnection
import pandas as pd
import numpy as np
from hydroDL import kPath

# write urls to a file

nodeUrl = 'https://esgf-node.llnl.gov/esg-search'
conn = SearchConnection(nodeUrl, distrib=False)
ctx = conn.new_context()
dictSearch = dict(project='CMIP6', activity_id='HighResMIP',
                  frequency='day', nominal_resolution='50 km',
                  variable='pr,tas,tasmin,tasmax',
                  variant_label='r1i1p1f1',
                  latest=True)

dictLst = [
    dict(source_id='CNRM-CM6-1-HR', variant_label='r2i1p1f2',
         experiment_id='hist-1950'),
    dict(source_id='CNRM-CM6-1-HR', variant_label='r2i1p1f2',
         experiment_id='highres-future'),

    dict(source_id='EC-Earth3P', variant_label='r3i1p2f1',
         experiment_id='hist-1950'),
    dict(source_id='EC-Earth3P', variant_label='r3i1p2f1',
         experiment_id='highres-future'),

    dict(source_id='EC-Earth3P-HR', variant_label='r1i1p2f1',
         experiment_id='hist-1950'),
    dict(source_id='EC-Earth3P-HR', variant_label='r1i1p2f1',
         experiment_id='highres-future'),

    dict(source_id='HadGEM3-GC31-HM', experiment_id='hist-1950'),
    dict(source_id='HadGEM3-GC31-HM', experiment_id='highres-future'),

    dict(source_id='MPI-ESM1-2-XR', experiment_id='hist-1950'),
    dict(source_id='MPI-ESM1-2-XR', experiment_id='highres-future'),

    dict(source_id='FGOALS-f3-H', variable='pr,tas', nominal_resolution='25 km',
         data_node='esgf-data1.llnl.gov', experiment_id='highres-future'),
    dict(source_id='FGOALS-f3-H', variable='pr,tas', nominal_resolution='25 km',
         experiment_id='hist-1950')
]
# write urlFiles
# f = open('downloadCMIP6', 'w')
for dd in dictLst:
    dictTemp = dictSearch.copy()
    dictTemp.update(dd)
    ctx = conn.new_context(**dictTemp)
    if ctx.hit_count != len(dictTemp['variable'].split(','))*2:
        print('something wrong', ctx.hit_count)
        print(dd)
    if dd['experiment_id'] == 'hist-1950':
        fileName = '{}-hist'.format(dd['source_id'])
    elif dd['experiment_id'] == 'highres-future':
        fileName = '{}-future'.format(dd['source_id'])
    urlFile = os.path.join(kPath.dirCode, 'app', 'data',
                           'cmip', 'urlFile', fileName)
    # 'a' is much faster than 'x'
    # manually delete existing files
    f = open(urlFile, 'a')
    for i, rr in enumerate(ctx.search()):
        files = rr.file_context().search()
        print(i)
        for j, ff in enumerate(files):
            print(j)
            _ = f.write(ff.download_url+'\n')
    f.close()
