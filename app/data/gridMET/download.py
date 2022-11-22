import hydroDL.data.gridMET.download as download

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = list(range(1979, 2021))

for var in varLst:
    for yr in yrLst:
        download.single(var, yr)
