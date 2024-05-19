import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import utils
from hydroDL.post import mapplot, axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainBasin
import math
import torch
from torch import nn
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range
import torch.optim as optim
from hydroDL import kPath
import torch.optim.lr_scheduler as lr_scheduler
import dill

rho = 45
dataName = "singleDaily-modisgrid"
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)
df.varX

dataName = "singleDaily-nadgrid"
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)
df.varX