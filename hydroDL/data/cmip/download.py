import urllib
import urllib.request
import time
from hydroDL import kPath
import os
import argparse


def byUrlFile(urlFile, outFolder, re=False):
    print('downloading {}'.format(urlFile), flush=True)
    with open(urlFile) as f:
        urlLst = f.read().splitlines()
    for url in urlLst:
        fileName = url.split('/')[-1]
        outFile = os.path.join(outFolder, fileName)
        if re is True or not os.path.exists(outFile):
            t0 = time.time()
            urllib.request.urlretrieve(url, outFile)
            print('{} {}'.format(fileName, t0-time.time()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', dest='urlFile', type=str)
    parser.add_argument('-O', dest='outFolder', type=str,
                        default=os.path.join(kPath.dirRaw, 'CMIP6'))
    args = parser.parse_args()
    byUrlFile(args.urlFile, args.outFolder)
