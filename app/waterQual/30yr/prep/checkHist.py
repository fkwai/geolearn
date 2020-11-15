from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import numpy as np
from hydroDL.data import usgs, gageII, gridMET, ntn

dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)

for code in sorted(usgs.newC[1:]):
    ic = wqData.varC.index(code)
    data = wqData.c[:, ic]
    data = data[~np.isnan(data)]
    data2 = utils.rmExt(data, p=5)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(data, density=True, bins=50)
    axes[1, 0].hist(np.log(data+1), density=True, bins=50)
    axes[0, 1].hist(data2, density=True, bins=50)
    axes[1, 1].hist(np.log(data2+1), density=True, bins=50)
    shortName = usgs.codePdf.loc[code]['shortName']
    fig.suptitle('{} {}'.format(code, shortName))
    fig.show()
    fig.savefig(code+'.png')
