import pandas as pd
import numpy as np


def readUsgsText(fileName, dataType=None):
    with open(fileName) as f:
        k = 0
        line = f.readline()
        while line[0] == "#":
            line = f.readline()
            k = k + 1
        headLst = line.split('\t')[:-1]
        typeLst = f.readline().split('\t')[:-1]
    pdf = pd.read_table(fileName, header=k).drop(0)
    dictUsgs = dict(s=str, d=str, n=np.float)
    dictType = dict()
    for h, t in zip(headLst, typeLst):
        dictType[h] = dictUsgs[t[-1]]
    out = pdf.astype(dictType)
    # modify
    if dataType == 'dailyTS':
        out = refineDailyTS(out)
    elif dataType == 'sample':
        out = refineSample(out)
    return out


def refineDailyTS(pdf):
    # rename observation fields
    headLst = pdf.columns.tolist()
    for i, head in enumerate(headLst):
        temp = head.split('_')
        if temp[0].isdigit():
            if len(temp) == 3:
                headLst[i] = temp[1]
                pdf[head] = pdf[head].astype(np.float)
            else:
                headLst[i] = temp[1] + '_cd'
    pdf.columns = headLst
    # time field
    pdf['datetime'] = pd.to_datetime(pdf['datetime'], format='%Y-%m-%d')
    return pdf


def refineSample(pdf):
    # rename observation fields
    headLst = pdf.columns.tolist()
    for i, head in enumerate(headLst):
        if head[1:].isdigit():
            if head.startswith('p'):
                headLst[i] = head[1:]
                pdf[head] = pdf[head].astype(np.float)
            else:
                headLst[i] = head[1:] + '_cd'
    pdf.columns = headLst
    # time field
    temp = pdf['sample_dt'] + ' ' + pdf['sample_tm']
    pdf['datetime'] = pd.to_datetime(temp, format='%Y-%m-%d %H:%M')
    return pdf
