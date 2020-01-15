import numpy as np

def ecoReg_ind(codeReg, codeLst):
    if type(codeReg) is list:
        l1 = codeReg[0]
        l2 = codeReg[1]
        l3 = codeReg[2]
    elif type(codeReg) is str:
        l1 = int(codeReg[0:2])
        l2 = int(codeReg[2:4])
        l3 = int(codeReg[4:6])    
    if l2 == 0:
        ind = np.where((codeLst[:, 0] == l1))[0]
    else:    
        if l3 == 0:
            ind = np.where((codeLst[:, 0] == l1) & (
                codeLst[:, 1] == l2))[0]
        else:
            ind = np.where((codeLst[:, 0] == l1) & (
                codeLst[:, 1] == l2) & (codeLst[:, 2] == l3))[0]
    return ind
