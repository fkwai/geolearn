import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np

# my data
outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabAll = pd.read_csv(outFile)
tabAll['Date'] = pd.to_datetime(tabAll['Date'], format='%Y-%m-%d')
tab1=tabAll[tabAll['Date']<'2019-03-01']
c1=tab1.groupby('Site').count()['Percent'].values

# old data all
fileOld='/home/kuai/GitHUB/lfmc_from_sar/input_data/lfmc_training_samples_updated_2019-04-29.csv'
tabOld=pd.read_csv(fileOld)
tabOld['date'] = pd.to_datetime(tabOld['date'], format='%Y-%m-%d')
tab2=tabOld[tabOld['date']>='2015-01-01']
tab2=tab2[~(tab2['state']=='AK')]
c2=tab2.groupby('site').count()['lfmc'].values

# old data train
dataFile = '/home/kuai/GitHUB/lfmc_from_sar/input_data/lstm_input_data_pure+all_same_28_may_2019_res_SM_gap_3M'
data = pd.read_pickle(dataFile)
c3=data.groupby('site').count()['percent(t)'].values