import argparse
import os
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
from hydroDL.data import usgs
from hydroDL import kPath
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', dest='dataName', type=str)
    parser.add_argument('-T', dest='trainSet', type=str)
    args = parser.parse_args()
    testSet = 'all'
    yW = WRTDS.testWRTDS(args.dataName, args.trainSet, testSet, usgs.varC)
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
    fileName = '{}-{}-{}'.format(args.dataName, args.trainSet, testSet)
    np.savez_compressed(os.path.join(dirRoot, fileName), yW)
