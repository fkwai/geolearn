import os
import socket
import collections
import imp

__all__ = ['classDB', 'funDB', 'classLSTM', 'funLSTM', 'kPath']

print('load rnnSMAP')

#################################################
# initialize data / out path of rnnSMAP


def initPath():
    hostName = socket.gethostname()
    if hostName == 'smallLinux':
        dirDB = os.path.join(
            os.path.sep, 'mnt', 'sdc', 'rnnSMAP', 'Database_SMAPgrid')
        dirOut = os.path.join(
            os.path.sep, 'mnt', 'sdb', 'rnnSMAP', 'Output_SMAPgrid')
        dirResult = os.path.join(
            os.path.sep, 'mnt', 'sdb', 'rnnSMAP', 'Result_SMAPgrid')
        os.environ[
            'PROJ_LIB'] = r'/home/kxf227/anaconda3/pkgs/proj4-5.2.0-he6710b0_1/share/proj/'
    elif hostName == 'AW-m17':
        dirDB = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Database_SMAPgrid')
        dirOut = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Model_SMAPgrid')
        dirResult = os.path.join(os.path.sep, 'D:', 'rnnSMAP',
                                 'Result_SMAPgrid')
        os.environ[
            'PROJ_LIB'] = r'C:\Users\geofk\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
    kPath = collections.OrderedDict(
        DB_L3_CONUS=os.path.join(dirDB, 'Daily_L3_CONUS'),
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        DB_L4_CONUS=os.path.join(dirDB, 'Daily_L4_CONUS'),
        DB_L4_NA=os.path.join(dirDB, 'Daily_L4_NA'),
        Out_L3_CONUS=os.path.join(dirOut, 'L3_CONUS'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        Out_L4_CONUS=os.path.join(dirOut, 'L4_CONUS'),
        Out_L4_NA=os.path.join(dirOut, 'L4_NA'),
        OutSigma_L3_NA=os.path.join(dirOut, 'L3_NA_sigma'),
        dirResult=dirResult)
    return kPath


kPath = initPath()

from . import kuaiLSTM
from . import classDB
from . import funDB
from . import classLSTM
from . import funLSTM
from . import classPost
from . import funPost
from . import funWeight


def reload():
    imp.reload(classDB)
    imp.reload(funDB)
    imp.reload(classLSTM)
    imp.reload(funLSTM)
    imp.reload(classPost)
    imp.reload(funPost)
    imp.reload(kuaiLSTM)
    imp.reload(funWeight)
