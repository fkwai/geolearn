from hydroDL.data import dbBasin

# DF = dbBasin.DataFrameBasin('allCQ')
DF = dbBasin.DataFrameBasin('Q90new')

q = DF.q[:, :, 0]
