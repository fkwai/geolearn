import pandas as pd
import numpy as np

# fileName = r'C:\Users\geofk\work\waterQuality\USGS\dailyTS\08075400'


def readUsgsText(fileName, dataType=None):
    with open(fileName) as f:
        k = 0
        line = f.readline()
        while line[0] == "#":
            line = f.readline()
            k = k + 1
        headLst = line[:-1].split('\t')
        typeLst = f.readline()[:-1].split('\t')
    pdf = pd.read_table(fileName, header=k, dtype=str).drop(0)
    for i, x in enumerate(typeLst):
        if x[-1] == 'n':
            pdf[headLst[i]] = pd.to_numeric(pdf[headLst[i]], errors='coerce')
        if x[-1] == 'd':
            pdf[headLst[i]] = pd.to_datetime(pdf[headLst[i]], errors='coerce')
    # modify
    if dataType == 'dailyTS':
        out = refineDailyTS(pdf)
    elif dataType == 'sample':
        out = refineSample(pdf)
    else:
        out = pdf
    return out


def refineDailyTS(pdf):
    # rename observation fields
    headLst = pdf.columns.tolist()
    for i, head in enumerate(headLst):
        temp = head.split('_')
        if temp[0].isdigit():
            if len(temp) == 3:
                headLst[i] = temp[1] + '_' + temp[2]
                pdf[head] = pdf[head].astype(np.float)
            else:
                headLst[i] = temp[1] + '_' + temp[2] + '_cd'
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
