
import argparse
import os
from hydroDL.master import basinFull

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', dest='masterName', type=str)
    args = parser.parse_args()
    basinFull.trainModelTS(args.masterName)
