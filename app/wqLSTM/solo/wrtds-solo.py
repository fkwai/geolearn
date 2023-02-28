from hydroDL.app.waterQuality import WRTDS

codeLst = ['00618', '00915', '00955']
for code in codeLst:
    dataName = '{}-{}'.format(code, 'B200')
    trainSet = 'rmYr5b0'
    yW = WRTDS.testWRTDS(dataName, trainSet, 'all', [code])
