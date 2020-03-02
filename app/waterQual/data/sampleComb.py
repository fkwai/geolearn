"""
Look at read C data after 1979-01-01, summarize sample combinations count
"""

# summarize relation between variables
dictSum = dict()
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfC = usgs.readSample(siteNo, codeLst=waterQuality.codeLst,
                          startDate=pd.datetime(1979, 1, 1))
    dfC.to_csv(os.path.join(kPath.dirData, 'USGS',
                            'sample', 'csv', siteNo+'.csv'))
    for k, row in dfC.iterrows():
        temp = dfC.columns[~pd.isna(row)].tolist()
        dictName = '-'.join(temp)
        if dictName not in dictSum:
            dictSum[dictName] = 1
        else:
            dictSum[dictName] = dictSum[dictName]+1
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')

tab = pd.DataFrame.from_dict(dictSum, orient='index')
tab = tab.sort_values(0, ascending=False)
tab.to_csv(os.path.join(dirInv, 'codeCombCount'), header=False)