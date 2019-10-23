import os
import socket
import collections
print('loading package hydroDL')


def initPath():
    """initial shortcut for some import paths
    """
    hostName = socket.gethostname()
    if hostName == 'smallLinux':
        dirDB = os.path.join(os.path.sep, 'mnt', 'sdc', 'rnnSMAP',
                             'Database_SMAPgrid')
        dirOut = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                              'Model_SMAPgrid')
        dirResult = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                                 'Result_SMAPgrid')
        os.environ[
            'PROJ_LIB'] = r'/home/kxf227/anaconda3/pkgs/proj4-5.2.0-he6710b0_1/share/proj/'
    elif hostName == 'AW-m17':
        dirDB = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Database_SMAPgrid')
        dirOut = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Model_SMAPgrid')
        dirResult = os.path.join(os.path.sep, 'D:', 'rnnSMAP',
                                 'Result_SMAPgrid')
        os.environ[
            'PROJ_LIB'] = r'C:\Users\geofk\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
    else:
        dirDB = '/'
        dirOut = '/'
        dirResult = '/'

    pathSMAP = collections.OrderedDict(
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        outTest=os.path.join(dirOut, 'Test'),
        dirDB=dirDB,
        dirOut=dirOut,
        dirResult=dirResult)

    pathCamels = collections.OrderedDict(
        DB=os.path.join(os.path.sep, 'mnt', 'sdb', 'Data', 'Camels'),
        Out=os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnStreamflow'))
    return pathSMAP, pathCamels

pathSMAP, pathCamels = initPath()


# from . import utils
# from . import data
# from . import model
# from . import post