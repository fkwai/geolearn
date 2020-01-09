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
    if k == 0:
        return None

    pdf = pd.read_table(fileName, header=k, dtype=str).drop(0)
    for i, x in enumerate(typeLst):
        if x[-1] == 'n':
            pdf[headLst[i]] = pd.to_numeric(pdf[headLst[i]], errors='coerce')
        if x[-1] == 'd':
            pdf[headLst[i]] = pd.to_datetime(pdf[headLst[i]], errors='coerce')
    # modify - only rename head or add columns, will not modify values
    if dataType == 'dailyTS':
        out = refineDailyTS(pdf)
    elif dataType == 'sample':
        out = refineSample(pdf)
    elif dataType == 'streamflow':
        out = refineStreamflow(pdf)
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
    pdf['date'] = pd.to_datetime(pdf['datetime'], format='%Y-%m-%d')
    return pdf


def refineStreamflow(pdf):
    # pick the longest average Q field
    headLst = pdf.columns.tolist()
    tempS = [head for head in headLst if head[-1] == '3']
    tempN = [pdf[head].isna().sum() for head in headLst if head[-1] == '3']
    ind = tempN.index(min(tempN))
    code = int(tempS[ind].split('_')[0])
    # searched and no code of leading zero
    dictRename = {'{}_00060_{:05n}'.format(
        code-2+x, x+1): '00060_{:05n}'.format(x+1) for x in range(3)}
    pdf = pdf.rename(columns=dictRename)
    # time field
    pdf['date'] = pd.to_datetime(pdf['datetime'], format='%Y-%m-%d')
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
    # time field - not work for nan time, use date for current
    # temp = pdf['sample_dt'] + ' ' + pdf['sample_tm']
    # pdf['datetime'] = pd.to_datetime(temp, format='%Y-%m-%d %H:%M')
    pdf['date'] = pd.to_datetime(pdf['sample_dt'], format='%Y-%m-%d')
    return pdf
