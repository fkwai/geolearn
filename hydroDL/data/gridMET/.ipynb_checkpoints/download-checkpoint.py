import urllib
import urllib.request
import time
from hydroDL import kPath
import os
import argparse

outFolder = os.path.join(kPath.dirRaw, 'gridMet')
urlPat = r'http://www.northwestknowledge.net/metdata/data/{}_{}.nc'


def single(var, yr, outFolder=outFolder, re=False):
    url = urlPat.format(var, yr)
    print('downloading {}'.format(url), flush=True)
    t0 = time.time()
    fileName = url.split('/')[-1]
    outFile = os.path.join(outFolder, fileName)
    urllib.request.urlretrieve(url, outFile)
    print('{} {}'.format(fileName, time.time()-t0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', dest='urlFile', type=str)
    parser.add_argument('-O', dest='outFolder', type=str,
                        default=outFolder)
    args = parser.parse_args()
