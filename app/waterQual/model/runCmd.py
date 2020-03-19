
import argparse
import os
from hydroDL.master import basins

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', dest='data', type=str)
    parser.add_argument('-O', dest='opt', type=str)
    args = parser.parse_args()

    basins.trainModelTS(args.data, 'first80', batchSize=[
                        None, 500], saveName='HBN_opt'+args.opt, optQ=int(args.opt))

