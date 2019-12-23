import os
import socket
import collections


def initPathSMAP(dirDB, dirOut, dirResult):
    pathSMAP = collections.OrderedDict(
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        outTest=os.path.join(dirOut, 'Test'),
        dirDB=dirDB,
        dirOut=dirOut,
        dirResult=dirResult)
    return pathSMAP


hostName = socket.gethostname()

if hostName == 'AW-m17':
    dirDB = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Database_SMAPgrid')
    dirOut = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Model_SMAPgrid')
    dirResult = os.path.join(os.path.sep, 'D:', 'rnnSMAP',
                             'Result_SMAPgrid')
    pathSMAP = initPathSMAP(dirDB, dirOut, dirResult)
    os.environ[
        'PROJ_LIB'] = r'C:\Users\geofk\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
elif hostName[:4] == 'icme':
    dirData = r'/home/kuaifang/Data/'
    dirWQ = r'/home/kuaifang/waterQuality/'
