# %% initial
from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
import torch
torch.cuda.set_device(2)


subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst1 = ['Local', 'CONUS']
caseLst2 = ['Forcing', 'Soilm']
saveFolder = os.path.join(pathSMAP['dirResult'], 'regionalization')
# k = [7, 8, 13]
# %%  load data and stat
kcLst = [7, 8, 13]
tRange = [20160401, 20180401]
statLst = list()
statRefLst = list()
for kc in kcLst:
    tempLst = list()    
    for k in range(1, 18):
        testName = subsetLst[kc-1]
        if k != kc:
            outName = 'ecoRegion{:02d}{:02d}_v2f1_Forcing'.format(kc, k)
        else:
            outName = 'ecoRegion{:02d}_v2f1_Forcing'.format(kc)
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion', outName)
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        temp = stat.statError(yp[:, :, 0], yt[:, :, 0])
        tempLst.append(temp)        
        if k == kc:
            statRefLst.append(temp)    
    statLst.append(tempLst)


# %% plot box
keyLst = stat.keyLst
ecoLst = ['{:02d}'.format(x) for x in range(1, 18)]
caseLst = ['{:02d}'.format(x) for x in [7, 8, 13]]

for k in range(len(caseLst)):
    dataBox = list()
    key = 'RMSE'
    for ii in range(len(ecoLst)):
        temp = list()
        temp.append(statLst[k][ii][key]-statRefLst[k][key])        
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox, ecoLst, caseLst,
                          title=key, figsize=(12, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'ecoRegionComb_box_' + str(caseLst[k]))
    fig.savefig(saveFile)
