
import argparse
import os
from hydroDL.master import basins

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', dest='masterName', type=str)
    args = parser.parse_args()
    basins.trainModelTS(args.masterName)
