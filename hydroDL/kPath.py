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
    dirData = r'C:\Users\geofk\work\database'
    dirRaw = r'F:\dataRaw'
    dirUsgs = r'F:\USGS'
    dirWQ = r'C:\Users\geofk\work\waterQuality'
    dirCode = r'C:\Users\geofk\work\GitHUB\geolearn'
    # dirWQ = r'C:\Users\geofk\work\example\waterQuality'
    # dirData = r'C:\Users\geofk\work\example\data'

elif hostName[:4] == 'icme':
    host = 'icme'
    dirData = r'/home/kuaifang/Data/'
    dirWQ = r'/home/kuaifang/waterQuality/'
    dirJob = r'/home/kuaifang/jobs/'
elif hostName[:2] == 'sh':
    host = 'sherlock'
    dirData = r'/oak/stanford/schools/ees/kmaher/Kuai/Data/'
    dirRaw = r'/oak/stanford/schools/ees/kmaher/DataRaw/'
    dirUsgs = r'/oak/stanford/schools/ees/kmaher/Kuai/dbUSGS'
    dirWQ = r'/oak/stanford/schools/ees/kmaher/Kuai/waterQuality/'
    dirJob = r'/scratch/users/kuaifang/jobs/'
    dirCode = r'/home/users/kuaifang/GitHUB/geolearn'
    dirVeg=r'/oak/stanford/schools/ees/kmaher/Kuai/VegetationWater/data/'
elif hostName == 'mini':
    host = 'mini'
    dirData = r'/home/kuai/work/Data/'
    dirUsgs = r'/home/kuai/work/dbUSGS/'
    dirRaw = r'/mnt/sda/dataRaw/'
    dirWQ = r'/home/kuai/work/waterQuality/'    
    dirCode = r'/home/kuai/GitHUB/geolearn'
    dirVeg = r'/home/kuai/work/VegetationWater/data/'

