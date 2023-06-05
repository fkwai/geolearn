from hydroDL.app.waterQuality import WRTDS
code = '00955'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
dataName = '{}-B200'.format(code)
yW2 = WRTDS.testWRTDS(dataName, trainSet, testSet, [code])
