import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET


matCodeUsgs = [
    ['00010', "Temperature, water", 'Temp', 'Physical', 'deg C',
        'Physical', "Temperature, water, degrees Celsius"],
    ['00930', 'Sodium', 'Na', 'Rock', 'mg/l', "Inorganics, Major, Metals",
        "Sodium, water, filtered, milligrams per liter"],
    ['00095', 'Specific' 'conductance', 'kappa', 'Physical', 'uS/cm @25C', 'Physical',
        "Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius"],
    ['00400', 'pH', 'pH', 'Physical', '[]', 'Physical',
        "pH, water, unfiltered, field, standard units"],
    ['80154', 'Suspended sediment concentration (SSC)', 'SSC', 'Physical', 'mg/l',
     'Sediment', "Suspended sediment concentration, milligrams per liter"],
    ['00940', 'Chloride', 'Cl', 'Rock', 'mg/l', "Inorganics, Major, Non-metals",
        "Chloride, water, filtered, milligrams per liter"],
    ['00945', 'Sulfate', 'SO4', 'Rock', 'mg/l', "Inorganics, Major, Non-metals",
        "Sulfate, water, filtered, milligrams per liter"],
    ['00405', 'Carbon' 'dioxide', 'CO2', 'Bio', 'mg/l', "Inorganics, Major, Non-metals",
        "Carbon dioxide, water, unfiltered, milligrams per liter"],
    ['70303', 'Total dissolved solids,TDS', 'Physical', 'tons/ac ft',
        'Physical', "Dissolved solids, water, filtered, short tons per acre-foot"],
    ['00915', 'Calcium', 'Ca', 'Rock', 'mg/l', "Inorganics, Major, Metals",
        "Calcium, water, filtered, milligrams per liter"],
    ['00300', 'Oxygen', 'O', 'Bio', 'mg/l', "Inorganics, Major, Non-metals",
        "Dissolved oxygen, water, unfiltered, milligrams per liter"],
    ['00925', 'Magnesium', 'Mg', 'Rock', 'mg/l', "Inorganics, Major, Metals",
        "Magnesium, water, filtered, milligrams per liter"],
    ['00660', 'Phosphate', 'PO4', 'Bio', 'mg/l', 'Nutrient',
        "Orthophosphate, water, filtered, milligrams per liter as PO4"],
    ['00955', 'Silica', 'SiO2', 'Rock', 'mg/l', "Inorganics, Major, Non-metals",
        "Silica, water, filtered, milligrams per liter as SiO2"],
    ['00935', 'Potassium', 'K', 'Bio', 'mg/l', "Inorganics, Major, Metals",
        "Potassium, water, filtered, milligrams per liter"],
    ['00618', 'Nitrate', 'NO3', 'Bio', 'mg/l as N', 'Nutrient',
        "Nitrate, water, filtered, milligrams per liter as nitrogen"],
    ['00950', 'Fluoride', 'F', 'Rock', 'mg/l', "Inorganics, Major, Non-metals",
        "Fluoride, water, filtered, milligrams per liter"],
    ['00410', 'Alkalinity', 'Alk', 'Rock', 'mg/l CaCO3', "Inorganics, Major, Non-metals",
        "Acid neutralizing capacity, water, unfiltered, fixed endpoint (pH 4.5) titration, field, milligrams per liter as calcium carbonate"],
    ['00681', 'Organic carbon', 'C', 'Bio', 'mg/l', "Organics, other",
        "Organic carbon, water, filtered, milligrams per liter"],
    ['00600', 'N total', 'Bio', 'mg/l', 'Nutrient', "Total nitrogen [nitrate + nitrite + ammonia + organic-N], water, unfiltered, milligrams per liter",
        "Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)"],
    ['00440', 'Bicarbonate', 'CO3', 'Bio', 'mg/l', "Inorganics, Major, Non-metals",
        "Bicarbonate, water, unfiltered, fixed endpoint (pH 4.5) titration, field, milligrams per liter"],
    ['00665', 'Phosphorus', 'P', 'Rock', 'mg/l', 'Nutrient',
        "Phosphorus, water, unfiltered, milligrams per liter as phosphorus"],
    ['00605', 'Organic nitrogen', 'N organic', 'Bio', 'mg/l', 'Nutrient',
        "Organic nitrogen, water, unfiltered, milligrams per liter as nitrogen"],
    ['71846', 'Ammonia and ammonium', 'NH3 NH4', 'Bio', 'mg/l NH4', 'Nutrient', "Ammonia (NH3 + NH4+), water, filtered, milligrams per liter as NH4"]]
codePdf = pd.DataFrame(matCodeUsgs, columns=['code', 'srsName', 'shortName', 'group', 'unit', 'usgsGroup','fullName']).set_index('code')
codeLst = list(codePdf.index)


def wrapData(caseName, siteNoLst, *, rho=365, nFill=5, varC=usgs.lstCodeSample, varG=gageII.lstWaterQuality, targetQ=False):
    """ wrap up input and target data for the model,as:
    x=[nT,nP,nX]
    y=[nP,nY]
    c=[nP,nC]
    where nP is number of time series
    Arguments:
        caseName {str} -- name of current data case
        siteNoLst {list} -- list of USGS site
    Keyword Arguments:
        rho {int} -- [description] (default: {365})
        nFill {int} -- max number of continous nan to interpolate in input data (default: {5})
        varC {list} -- list of water quality code to learn (default: {usgs.lstCodeSample})
        varG {list} -- list of constant variables in gageII (default: {gageII.lstWaterQuality})
        targetQ {bool} -- True: streamflow is target; False: streamflow is input (default: {False})
    """
    # caseName = 'temp'
    # siteNoLst = ['06228350', '08082180', '04213000', '06623800', '08081200']
    # rho = 365
    # nFill = 5
    # varC = usgs.lstCodeSample
    # varG = gageII.lstWaterQuality
    # targetQ = False

    # add a start date to improve efficiency.
    startDate = pd.datetime(1979, 1, 1)

    # gageII
    tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
    tabG = gageII.updateCode(tabG)

    # read data and merge to: x=[nT,nP,nX], y=[nP,nY], c=[nP,nC]
    xLst = list()
    yLst = list()
    cLst = list()
    infoLst = list()
    for i, siteNo in enumerate(siteNoLst):
        t0 = time.time()

        dfC = usgs.readSample(siteNo, codeLst=varC, startDate=startDate)
        dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
        dfF = gridMET.readBasin(siteNo)

        t1 = time.time()
        for k in range(len(dfC)):
            yt = dfC.index[k]
            if yt-pd.Timedelta(days=rho) < startDate:
                continue
            dfX = pd.DataFrame({'date': pd.date_range(
                yt-pd.Timedelta(days=rho-1), yt)}).set_index('date')
            dfX = dfX.join(dfQ)
            dfX = dfX.join(dfF)
            dfX = dfX.interpolate(limit=nFill, limit_direction='both')
            if not dfX.isna().values.any():
                xLst.append(dfX.values)
                yLst.append(dfC.iloc[k].values)
                cLst.append(tabG.loc[siteNo].values)
                infoLst.append(dict(siteNo=siteNo, date=yt))
        t2 = time.time()
        print('{} on site {} reading {:.3} processing {:.3}'.format(
            i, siteNo, t1-t0, t2-t1))
    x = np.stack(xLst, axis=-1).swapaxes(1, 2).astype(np.float64)
    y = np.stack(yLst, axis=-1).swapaxes(0, 1).astype(np.float64)
    c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float64)
    infoDf = pd.DataFrame(infoLst)

    saveFolder = os.path.join(kPath.dirWQ, 'tempData')
    saveName = os.path.join(saveFolder, caseName)
    np.savez(saveName, x=x, y=y, c=c)
    infoDf.to_csv(saveName+'.csv')
    dictData = dict(name=caseName, rho=rho, nFill=nFill,
                    varG=varG, varC=varC, targetQ=False,
                    siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)


def loadData(caseName):
    saveName = os.path.join(kPath.dirWQ, 'tempData', caseName)
    npzFile = np.load(saveName+'.npz')
    x = npzFile['x']
    y = npzFile['y']
    c = npzFile['c']
    info = pd.read_csv(saveName+'.csv', index_col=0, dtype={'siteNo': str})
    with open(saveName+'.json', 'r') as fp:
        dictData = json.load(fp)
    return dictData, info, x, y, c


def loadInfo(caseName):
    saveName = os.path.join(kPath.dirWQ, 'tempData', caseName)
    info = pd.read_csv(saveName+'.csv', index_col=0, dtype={'siteNo': str})
    with open(saveName+'.json', 'r') as fp:
        dictData = json.load(fp)
    return dictData, info


def divideTrain(info, ratio=0.8):
    # devide training and testing - last 20% as testing
    siteNoLst = info['siteNo'].unique().tolist()
    trainIndLst = list()
    testIndLst = list()
    for siteNo in siteNoLst:
        indLst = info[info['siteNo'] == siteNo].index.tolist()
        indSep = round(len(indLst)*0.8)
        trainIndLst = trainIndLst+indLst[:indSep]
        testIndLst = testIndLst+indLst[indSep:]
    return trainIndLst, testIndLst
