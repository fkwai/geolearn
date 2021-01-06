import numpy as np
import pandas as pd
import os

fileEco = r'C:\Users\geofk\work\map\ecoRegion\comb\ecoRegionLev2.csv'
tabEco = pd.read_csv(fileEco)

tabArea = tabEco.groupby(['NA_L2CODE']).sum()

tabArea['Shape_Area'].sum()