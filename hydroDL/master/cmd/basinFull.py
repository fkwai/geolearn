
import argparse
import os
from hydroDL.master import basinFull

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', dest='masterName', type=str)
    parser.add_argument('-R', dest='resumeOpt', type=int)
    args = parser.parse_args()
    if args.resumeOpt is None:
        basinFull.trainModel(args.masterName)
    else:
        basinFull.resumeModel(args.masterName, resumeOpt=args.resumeOpt)
