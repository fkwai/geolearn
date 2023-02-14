from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
from hydroDL import kPath, utils
import pickle

import torch
caseName='NY5-QFT2C-B15'
debugName = 'debug4-199-0209160140'
# debugName='debug15-127-0209113728'

fileDebug=os.path.join(basinFull.nameFolder(caseName),debugName)

import io
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

with open(fileDebug, 'rb') as fh:
    data = CPU_Unpickler(fh).load()      


xT=data['xT']
yT=data['yT']
lossFun=data['lossFun']
model=data['model']

yP = model(xT)
loss = lossFun(yP, yT)
loss.backward()
for name, param in model.named_parameters():    
    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        print(11)
        print(torch.isnan(param.grad))
        print(name)