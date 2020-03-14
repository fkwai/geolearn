
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from datetime import date
import warnings


from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit, trainTS
from hydroDL.master import basins


basins.trainModelTS('temp10', 'first80', batchSize=[10, 365])

# # predict - point-by-point
# yOut = trainTS.testModel(model, x, xc)
# q, c = wqData.transOut(yOut[:, :, :ny], yOut[-1, :, ny:], statY, statYC)
