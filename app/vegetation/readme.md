# summary of code flow

## Code repo

### where it is

- [general library](https://github.com/fkwai/geolearn/tree/master)

- [vegetation water](https://github.com/fkwai/geolearn/tree/master/app/vegetation)


### how to use

- change corresponding path in the [kPath.py](https://github.com/fkwai/geolearn/blob/master/hydroDL/kPath.py)

## Download data

### NFMD

- download data from [NFMD website](https://www.wfas.net/nfmd/public/index.php), [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/NFMD/download.py)

- transfer to a csv file named *NFMD.csv* [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/NFMD/raw2csv.py)

- extract site information to *NFMDsite.csv* [code] (https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/NFMD/siteInfo.py) and assigned siteID [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/NFMD/addSiteId.py)

- extract data with single species and after 2019 records [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/NFMD/screenSite.py)

### Remote Sensing
- download modis, landsat and sentinel1 using GEE [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/RS/download-all.py)

## warp data
- wrap all above data to a format named [dataFrame](https://github.com/fkwai/geolearn/blob/master/hydroDL/data/dbVeg.py)
- [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/data/wrap/daily/wrapScript.py)

## train model
- train and test attention model [code](https://github.com/fkwai/geolearn/blob/master/app/vegetation/attention/data.py)

