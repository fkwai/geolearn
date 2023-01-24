# upgrade code to read flags and save CSV
from hydroDL.data import usgs
from hydroDL import kPath
from hydroDL.app import waterQuality
import os
import pandas as pd

siteNo = '07060710'
dfC = usgs.readSample(siteNo, codeLst=usgs.codeLst, flag=True, csv=False)
