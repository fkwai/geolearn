import os
import pandas as pd
from hydroDL.utils.time import tRange2Array
from .read import *

varF = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp']
varQ = ['q', 'runoff']
varG = [
    'elev_mean',
    'slope_mean',
    'area_gages2',
    'frac_forest',
    'lai_max',
    'lai_diff',
    'dom_land_cover_frac',
    'dom_land_cover',
    'root_depth_50',
    'soil_depth_statsgo',
    'soil_porosity',
    'soil_conductivity',
    'max_water_content',
    'geol_1st_class',
    'geol_2nd_class',
    'geol_porostiy',
    'geol_permeability'
]

dictStatQ = {
    'runoff': 'log-norm',
    'q': 'log-norm'}
dictStatF = {
    'dayl': 'norm',
    'prcp': 'log-norm',
    'srad': 'norm',
    'tmax': 'norm',
    'tmin': 'norm',
    'vp': 'norm'}
dictStatG = {
    'elev_mean': 'log-norm',
    'slope_mean': 'log-norm',
    'area_gages2': 'log-norm',
    'frac_forest': 'norm',
    'lai_max': 'norm',
    'lai_diff': 'norm',
    'dom_land_cover_frac': 'norm',
    'dom_land_cover': 'norm',
    'root_depth_50': 'norm',
    'soil_depth_statsgo': 'norm',
    'soil_porosity': 'norm',
    'soil_conductivity': 'log-norm',
    'max_water_content': 'norm',
    'geol_1st_class': 'norm',
    'geol_2nd_class': 'norm',
    'geol_porostiy': 'norm',
    'geol_permeability': 'norm'}


def extractVarMtd(varLst):
    mtdLst = list()
    if varLst is None:
        mtdLst = None
    else:
        for var in varLst:
            if var in dictStatQ.keys():
                mtd = dictStatQ[var]
            elif var in dictStatF.keys():
                mtd = dictStatF[var]
            elif var in dictStatG.keys():
                mtd = dictStatG[var]
            else:
                raise Exception('Variable {} not found!'.format(var))
            mtdLst.append(mtd)
    return mtdLst
