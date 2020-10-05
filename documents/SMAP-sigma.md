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
To plot figures, required data path need to be set correspondingly by modify Line 23-26 of "rnnSMAP\\__init__.py":
~~~
dirDB = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Database_SMAPgrid')
dirOut = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Output_SMAPgrid')
dirResult = os.path.join(os.path.sep, 'D:', 'rnnSMAP', 'Result_SMAPgrid')
os.environ['PROJ_LIB'] = r'C:\Users\geofk\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
~~~

# Figures and corresponding code

- [Figure 1](app/paperSigma/CONUSv4_noise.py) Maps for LSTM model error and uncertainty qualities 
- [Figure 2](app/paperSigma/CONUS_conf.py) Calibration plots of error exceedance likelihood 
- [Figure 3](app/paperSigma/CONUSv4_noise.py) Test for noise-added observations