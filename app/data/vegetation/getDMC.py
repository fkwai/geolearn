import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot


DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
fileTry = os.path.join(DIR_VEG,'TRY', 'TRY-DMC.txt')
tabTry = pd.read_table(fileTry,encoding= 'unicode_escape')
tabTry['DataID'].unique()