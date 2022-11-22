
import json
import numpy as np
from numpy import double
import pandas as pd
import os
import hydroDL
from hydroDL.data import camels, usgs, dbBasin
import importlib
import matplotlib.pyplot as plt
from hydroDL.post import axplot
from hydroDL.utils.time import t2dt
from hydroDL.master import basinFull, slurm

importlib.reload(camels)
dfInfo = camels.dfInfo
dfG = camels.readAttr(varLst=camels.varG)
