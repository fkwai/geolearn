from .io import *
from .dataModel import *


def label2var(label):
    dictVar = dict(
        F=gridMET.varLst,
        Q=['runoff'],
        P=ntn.varLst,
        T=['datenum', 'sinT', 'cosT'],
        R=GLASS.varLst,
        C=usgs.newC)
    varLst = list()
    for x in label:
        varLst = varLst + dictVar[x]
    return varLst
