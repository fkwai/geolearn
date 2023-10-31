import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.master import basinFull
from hydroDL.master import slurm
import time

codeLst = usgs.varC

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/wqLSTM/B200/solo/testSoloEpCMD.py -C {} -L {}'

label='QFT2C'
for code in codeLst:
    slurm.submitJob('testEp {}'.format(code), cmdP.format(code), nH=2, nM=32)
