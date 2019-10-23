from hydroDL import pathCamels, master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
import numpy as np
import os
import scipy.stats as stats

gageflow = camels.readUsgs(camels.gageDict['id'])
# tempflow = gageflow.flatten()
# tempflow = tempflow[~np.isnan(tempflow)] # kick out Nan
# # get the distribution
# plt.figure()
# plt.hist(tempflow, bins=5000, density=True)
# fit_alpha, fit_loc, fit_beta = stats.gamma.fit(tempflow, floc=0)
# print(fit_alpha, fit_loc, fit_beta)
# xx = np.arange(tempflow.min()+0.1, tempflow.max(), 50)
# yy = stats.gamma.pdf(xx, a=fit_alpha, loc=fit_loc, scale=fit_beta)
# plt.plot(xx,yy)
# plt.show()
# logtest=np.log10(np.sqrt(tempflow)+1)
# plt.figure()
# plt.hist(logtest, bins=50)
# precipitation
prcpts = camels.readForcing(camels.gageDict['id'], ['prcp'])
prcp = prcpts.flatten()
prcp = prcp[~np.isnan(prcp)] # kick out Nan
plt.figure()
plt.hist(prcp, bins=100)
logtest=np.log10(np.sqrt(prcp+1))
plt.figure()
plt.hist(logtest, bins=100)

# # normalize with basin area
# basinarea = camels.readAttr(camels.gageDict['id'], ['area_gages2'])
# temparea = np.tile(basinarea, (1, gageflow.shape[1]))
# flowua = gageflow*0.0283168/(temparea*(10**6))*3600*24*365 # unit m/year
# tempflow = flowua.flatten()
# tempflow = tempflow[~np.isnan(tempflow)] # kick out Nan
# # get the distribution
# plt.figure()
# plt.hist(tempflow, bins=5000, density=True)
# plt.show()

# # normalize with precipitation
# basinarea = camels.readAttr(camels.gageDict['id'], ['area_gages2'])
# meanprep = camels.readAttr(camels.gageDict['id'], ['p_mean'])
# temparea = np.tile(basinarea, (1, gageflow.shape[1]))
# tempprep = np.tile(meanprep, (1, gageflow.shape[1]))
# flowua = (gageflow*0.0283168*3600*24)/((temparea*(10**6))*(tempprep*10**(-3))) # unit m/year
# tempflow = flowua.flatten()
# tempflow = tempflow[~np.isnan(tempflow)] # kick out Nan
# # get the distribution
# plt.figure()
# plt.hist(tempflow, bins=5000, density=True)
# plt.show()
# logtest=np.log10(np.sqrt(tempflow)+0.1)
# plt.figure()
# plt.hist(logtest, bins=50,density=True)

