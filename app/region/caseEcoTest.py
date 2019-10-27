import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL import master


caseLst = ['080305', '080300', '080305+080400+080500', '080305+090200',
           '080305+090300', '080305+090400', '080305+100200', '080305+060200']

# test
tRange = [20150401, 20160401]
subsetPattern = 'ecoRegionL3_{}_v2f1'
testName = subsetPattern.format(caseLst[0])

case = caseLst[0]
outName = subsetPattern.format(case) + '_Forcing'
out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionL3', outName)
df, yp, yt = master.test(out, tRange=tRange, subset=testName)
