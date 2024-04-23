import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data_file='/home/kuai/GitHUB/lfmc_from_sar/input_data/lstm_input_data_pure+all_same_28_may_2019_res_SM_gap_3M'
int_lag=6
dataset= pd.read_pickle(data_file)

DROPCROPS = True
lc_dict = {14: 'crop',
            20: 'crop',
            30: 'crop',
            50: 'closed broadleaf deciduous',
            70: 'closed needleleaf evergreen',
            90: 'mixed forest',
            100:'mixed forest',
            110:'shrub/grassland',
            120:'grassland/shrubland',
            130:'closed to open shrub',
            140:'grass',
            150:'sparse vegetation',
            160:'regularly flooded forest'}
TRAINRATIO = 0.70
FOLDS = 3


microwave_inputs = ['vv','vh']
optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']
#optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv','vari','ndii']
mixed_inputs =  ['vv_%s'%den for den in optical_inputs] + ['vh_%s'%den for den in optical_inputs] + ['vh_vv']
dynamic_inputs = microwave_inputs + optical_inputs + mixed_inputs
static_inputs = ['slope', 'elevation', 'canopy_height','forest_cover',\
                    'silt', 'sand', 'clay']

all_inputs = static_inputs+dynamic_inputs
inputs = all_inputs

def split_train_test(dataset, inputs = None, int_lag = None, CV = False, fold = None, FOLDS = FOLDS):

    if DROPCROPS:
        crop_classes = [item[0] for item in lc_dict.items() if item[1] == 'crop']
        dataset = dataset.loc[~dataset['forest_cover(t)'].isin(crop_classes)]
    # integer encode forest cover
    encoder = LabelEncoder()
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
    cols = list(dataset.columns.values)
    for col in ['percent(t)','site','date']:
        cols.remove(col)
    cols = ['percent(t)','site','date']+cols
    dataset = dataset[cols]
    dataset['forest_cover(t)'] = encoder.fit_transform(dataset['forest_cover(t)'].values)
    for col in dataset.columns:
        if 'forest_cover' in col:
            dataset[col] = dataset['forest_cover(t)']
    # normalize features
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset.replace([np.inf, -np.inf], [1e5, 1e-5], inplace = True)
    scaled = scaler.fit_transform(dataset.drop(['site','date'],axis = 1).values)
    rescaled = dataset.copy()
    rescaled.loc[:,dataset.drop(['site','date'],axis = 1).columns] = scaled
    # reframed = series_to_supervised(rescaled, LAG,  dropnan = True)
    reframed = rescaled.copy()
    #### dropping sites with at least 7 training points
#    sites_to_keep = pd.value_counts(reframed.loc[reframed.date.dt.year<2018, 'site'])
#    sites_to_keep = sites_to_keep[sites_to_keep>=24].index
#    reframed = reframed.loc[reframed.site.isin(sites_to_keep)]
    
    print('[INFO] Dataset has %d sites'%len(reframed.site.unique()))
    ####    
    reframed.reset_index(drop = True, inplace = True)
    #print(reframed.head())
     
    # split into train and test sets
    # train = reframed.loc[reframed.date.dt.year<2018].drop(['site','date'], axis = 1)
    # test = reframed.loc[reframed.date.dt.year>=2018].drop(['site','date'], axis = 1)
    #### split train test as 70% of time series of each site rather than blanket 2018 cutoff
    train_ind=[]
    # for site in reframed.site.unique():
    #     sub = reframed.loc[reframed.site==site]
    #     sub = sub.sort_values(by = 'date')
    #     train_ind = train_ind+list(sub.index[:int(np.ceil(sub.shape[0]*TRAINRATIO))])
    if CV:
        for cover in np.sort(reframed['forest_cover(t)'].unique()):
            sub = reframed.loc[reframed['forest_cover(t)']==cover]
            sites = np.sort(sub.site.unique())
            
            if len(sites)<FOLDS:
                train_sites = sites
            else:
                train_sites_ind, _ = list(kf.split(sites))[fold]
                train_sites = sites[train_sites_ind]
                # break
            train_ind+=list(sub.loc[sub.site.isin(train_sites)].index)
        # print(len(train_ind)/reframed.shape[0])
    else:
        for cover in reframed['forest_cover(t)'].unique():
            sub = reframed.loc[reframed['forest_cover(t)']==cover]
            sites = sub.site.unique()
            train_sites = np.random.choice(sites, size = int(np.ceil(TRAINRATIO*len(sites))), replace = False)
            train_ind+=list(sub.loc[sub.site.isin(train_sites)].index)

        
    # sites = reframed.site.unique()
    # train_sites = np.random.choice(sites, size = int(np.ceil(TRAINRATIO*len(sites))), replace = False)
    # train_ind = reframed.loc[reframed.site.isin(train_sites)].index
    
    train = reframed.loc[train_ind].drop(['site','date'], axis = 1)
    test = reframed.loc[~reframed.index.isin(train_ind)].drop(['site','date'], axis = 1)
    train.sort_index(inplace = True)
    test.sort_index(inplace = True)
    #print(train.shape)
    #print(test.shape)
    # split into input and outputs
    train_X, train_y = train.drop(['percent(t)'], axis = 1).values, train['percent(t)'].values
    test_X, test_y = test.drop(['percent(t)'], axis = 1).values, test['percent(t)'].values
    # reshape input to be 3D [samples, timesteps, features]
    if inputs==None: #checksum
        inputs = all_inputs
    train_Xr = train_X.reshape((train_X.shape[0], int_lag+1, len(inputs)), order = 'A')
    test_Xr = test_X.reshape((test_X.shape[0], int_lag+1, len(inputs)), order = 'A')
    return dataset, rescaled, reframed, \
            train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
            scaler, encoder         

dataset, rescaled, reframed, \
    train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
    scaler, encoder = split_train_test(dataset, int_lag = int_lag)
import os
from hydroDL import kPath


def submitJob(jobName, cmdLine, nH=8, nM=16):
    jobFile = os.path.join(kPath.dirJob, jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        # fh.writelines('#SBATCH --qos=normal\n')
        # fh.writelines('#SBATCH --partition=owners\n')
        fh.writelines('#SBATCH --mail-type=ALL\n')        
        fh.writelines('#SBATCH --mail-user=kuaifang@stanford.edu\n')
        if kPath.host == 'icme':
            fh.writelines('source activate pytorch\n')
        elif kPath.host == 'sherlock':
            fh.writelines(
                'source /home/users/kuaifang/envs/pytorch/bin/activate\n')
        fh.writelines(cmdLine)
    os.system('sbatch {}'.format(jobFile))


def submitJobGPU(jobName, cmdLine, nH=8, nM=16):
    jobFile = os.path.join(kPath.dirJob, jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH -p gpu\n')
        fh.writelines('#SBATCH -G 1\n')
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        fh.writelines('#SBATCH --qos=normal\n')
        fh.writelines('#SBATCH -C "GPU_SKU:P100_PCIE|GPU_SKU:RTX_2080Ti|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2"')
        # fh.writelines('#SBATCH --partition=owners\n')
        fh.writelines('#SBATCH --mail-type=ALL\n')
        fh.writelines('#SBATCH --mail-user=kuaifang@stanford.edu\n')
        if kPath.host == 'icme':
            fh.writelines('source activate pytorch\n')
        elif kPath.host == 'sherlock':
            fh.writelines(
                'source /home/users/kuaifang/envs/pytorch/bin/activate\n')
        fh.writelines('hostname\n')
        fh.writelines('nvidia-smi -L\n')
        fh.writelines(cmdLine)
    os.system('sbatch {}'.format(jobFile))
