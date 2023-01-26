
from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json

siteNo='07182000'
saveFile='temp'
dfSite=gageII.readData(varLst=['STATE'])
state=dfSite.loc[siteNo]['STATE']
usgs.downloadDaily(siteNo, ['00060'], state, saveFile)

