import hydroDL.data.gridMET.download as download

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = list(range(2021, 2023))

for var in varLst:
    for yr in yrLst:
        download.single(var, yr)
