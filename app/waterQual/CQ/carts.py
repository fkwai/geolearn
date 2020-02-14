from sklearn.datasets import load_iris
from sklearn import tree
import os
import pandas as pd
import numpy as np
from hydroDL import kPath
from hydroDL.data import usgs, gageII

dirCQ = os.path.join(kPath.dirWQ, 'C-Q')
dfS = pd.read_csv(os.path.join(dirCQ, 'slope'), dtype={
    'siteNo': str}).set_index('siteNo')
dfN = pd.read_csv(os.path.join(dirCQ, 'nSample'), dtype={
                  'siteNo': str}).set_index('siteNo')
siteNoLst = dfS.index.tolist()
codeLst = dfS.columns.tolist()

pdf = gageII.readData(siteNoLst=siteNoLst).drop(columns=['STANAME'])

dropColLst = ['STANAME', 'WR_REPORT_REMARKS',
              'ADR_CITATION', 'SCREENING_COMMENTS']
pdf = gageII.updateCode(pdf)

clf = tree.DecisionTreeRegressor()
y = dfS['00955'].values
x = pdf.values
x[np.isnan(x)] = -99
ind=np.where(~np.isnan(y))
clf = clf.fit(x[ind], y[ind])
