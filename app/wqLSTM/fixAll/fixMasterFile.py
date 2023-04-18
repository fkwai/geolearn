from hydroDL import kPath
import os
from hydroDL.master import basinFull

modelFolder = os.path.join(kPath.dirWQ, 'modelFull')

outLst=os.listdir(modelFolder)

# fix a field in master file, and remove some fields
for out in outLst:
    dictMaster=basinFull.loadMaster(out)
    if 'resumeEpoch' in dictMaster.keys():
        dictMaster.pop('resumeEpoch')
    if 'overwrite' in dictMaster.keys():
        dictMaster.pop('overwrite')    
    if dictMaster['optim']=='AdaDelta':
        dictMaster['optim']='Adadelta'
    basinFull.wrapMaster(**dictMaster)

