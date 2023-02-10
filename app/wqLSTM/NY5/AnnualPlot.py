from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot

dataName='dbAll'
DF = dbBasin.DataFrameBasin(dataName)
