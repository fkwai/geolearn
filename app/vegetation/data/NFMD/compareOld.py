import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np

outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabAll = pd.read_csv(outFile)
tabAll['Date'] = pd.to_datetime(tabAll['Date'], format='%Y-%m-%d')

tab1=tabAll[tabAll['Date']<'2019-03-01']

fileOld='/home/kuai/GitHUB/lfmc_from_sar/input_data/lfmc_training_samples_updated_2019-04-29.csv'
tabOld=pd.read_csv(fileOld)
tabOld['date'] = pd.to_datetime(tabOld['date'], format='%Y-%m-%d')
tab2=tabOld[tabOld['date']>='2015-01-01']

tab2=tab2[~(tab2['state']=='AK')]
len(tab2['site'].unique())
