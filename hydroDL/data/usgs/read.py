import pandas as pd
import numpy as np


def readUsgsText(fileName, dataType=None):
    with open(fileName) as f:
        k = 0
        line = f.readline()
        while line[0] == "#":
            line = f.readline()
            k = k + 1
        # headLine=line.split('\t')[:-1]
        # typeLine=f.readline().split('\t')[:-1]
    skipRow = list(range(k))
    skipRow.append(k)
    pdf = pd.read_table(fileName, header=k).drop(0)
    # modify head
    if dataType == 'dailyTS':
        pdf = refineDailyTS(pdf)
    elif dataType == 'sample':
        pdf = refineSample(pdf)
    return pdf


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
