# Introduction
This document describes how to use this package to repeat the work presented in:

K. Fang, D. Kifer, K. Lawson, CP. Shen, Evaluating the potential and challenges of an uncertainty quantification method for long short-term memory models for soil moisture predictions

DOI will be added when available. 

The full package could be found in https://github.com/fkwai/geolearn

Please contact kuaifang@stanford.edu or cshen@engr.psu.edu for data. 

# Using of this code
## adding this code to the path
This folder needs to be added to the python path. We suggest adding this folder to the python developing path. In Linux it can be done by following shell command:
~~~
SITEDIR=$(python -m site --user-site)
echo "[path-of-this-folder]" > "$SITEDIR/[whatever].pth"
~~~

## modify data path
To plot figures, required data path need to be set correspondingly by modify Line 23-26 of [rnnSMAP package init code](/rnnSMAP/__init__.py):
~~~
dirDB = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Database_SMAPgrid')
dirOut = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Output_SMAPgrid')
dirResult = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Result_SMAPgrid')
os.environ['PROJ_LIB'] = r'C:\Users\geofk\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
~~~

# corresponding code
## Training and testing models
- [CONUS models](app/paperSigma/CONUS_temp_dr.py) Models trained over COUNS, presented in result 3.1, figure 1&2.
- [CONUS models with added noise](app/paperSigma/int_noise.py) CONUS model with noisy SMAP target, presented in result 3.2, figure 3.
- [Single eco-regions models](app/paperSigma/eco_single.py) Individual models for each level-2 eco-ecoregions, present in result 3.3, figure 4&5.
- [Combined eco-regions models](app/paperSigma/eco_comb.py) Models for regions that combined from level-2 eco-ecoregions, present in result 3.3, figure 6.
## Ploting figures 
- [Figure 1](app/paperSigma/CONUSv4_noise.py) Maps for LSTM model error and uncertainty qualities 
- [Figure 2](app/paperSigma/CONUS_conf.py) Calibration plots of error exceedance likelihood 
- [Figure 3](app/paperSigma/CONUSv4_noise.py) Test for noise-added observations
- [Figure 4](app/paperSigma/eco_single.py) CONUS Sigma_MC output from regional models
- [Figure 5](app/paperSigma/eco_single_box.py) Test on similar and dissimilar regions
- [Figure 6](app/paperSigma/eco_sigmabin.py) Test on uniform and comprehensive  regions
