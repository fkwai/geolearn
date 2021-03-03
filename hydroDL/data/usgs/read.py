import os
import pandas as pd
import numpy as np
from hydroDL import kPath

# fileName = r'C:\Users\geofk\work\waterQuality\USGS\dailyTS\08075400'

__all__ = ['readSample', 'readStreamflow', 'readUsgsText', 'removeFlag']


def readSample(siteNo, codeLst, startDate=None, csv=True, flag=0):
    """read USGS sample data, did:
    1. extract data of interested code and date
    2. average repeated daily observation
    Arguments:
        siteNo {str} -- site number
    Keyword Arguments:
        codeLst {list} -- usgs code of interesting fields (default: {sampleCodeLst})
        startDate {date} -- start date (default: {None})
        flag {int} -- 0 no flag; 1 str flag; 2 num flag 
    Returns:
        pandas.DataFrame -- [description]
    """
    if csv is False:
        fileC = os.path.join(kPath.dirData, 'USGS', 'sample', siteNo)
        dfC = readUsgsText(fileC, dataType='sample')
        if startDate is not None:
            dfC = dfC[dfC['date'] >= startDate]
        dfC = dfC.set_index('date')
        codeSel = list(set(codeLst) & set(dfC.columns.tolist()))
        codeSel_cd = [code + '_cd' for code in codeSel]
        dfC = dfC[codeSel+codeSel_cd].dropna(how='all')
        if len(dfC) == 0:
            return None if flag == 0 else (None, None)
        dfC1 = dfC[codeSel]
        dfC2 = dfC[codeSel_cd]
        bx = dfC1.notna().values & dfC2.isna().values
        dfC2[bx] = 'x'
        dfC2 = dfC2.fillna('')
        bDup = dfC.index.duplicated(keep=False)
        indUni = dfC.index[~bDup]
        indDup = dfC.index[bDup].unique()
        indAll = dfC.index.unique()
        dfO1 = pd.DataFrame(index=indAll, columns=codeSel)
        dfO2 = pd.DataFrame(index=indAll, columns=codeSel_cd)
        dfO1.loc[indUni] = dfC1.loc[indUni][codeSel]
        dfO2.loc[indUni] = dfC2.loc[indUni][codeSel_cd]
        for ind in indDup:
            temp1 = dfC1.loc[ind]
            temp2 = dfC2.loc[ind]
            for code in codeSel:
                if 'x' in temp2[code+'_cd'].tolist():
                    dfO1.loc[ind][code] = temp1[code][temp2[code+'_cd']
                                                      == 'x'].mean()
                    if temp2[code+'_cd'].tolist().count('x') > 1:
                        dfO2.loc[ind][code+'_cd'] = 'X'
                    else:
                        dfO2.loc[ind][code+'_cd'] = 'x'
                else:
                    dfO1.loc[ind][code] = temp1[code].mean()
                    dfO2.loc[ind][code+'_cd'] = ''.join(temp2[code+'_cd'])
    else:
        dirC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv')
        fileC1 = os.path.join(dirC, siteNo)
        if not os.path.exists(fileC1):
            return None if flag == 0 else (None, None)
        dfO1 = pd.read_csv(fileC1)
        dfO1['date'] = pd.to_datetime(dfO1['date'], format='%Y-%m-%d')
        dfO1 = dfO1.set_index('date')
        if flag > 0:
            fileC2 = os.path.join(dirC, siteNo+'_flag')
            dfO2 = pd.read_csv(fileC2)
            dfO2['date'] = pd.to_datetime(dfO2['date'], format='%Y-%m-%d')
            dfO2 = dfO2.set_index('date')
        if startDate is not None:
            dfO1 = dfO1[dfO1.index >= startDate]
            if flag > 0:
                dfO2 = dfO2[dfO2.index >= startDate]
    if flag > 0:
        if flag == 2:
            dfO3 = pd.DataFrame(
                index=dfO2.index, columns=dfO2.columns, dtype=int)
            dfO3[(dfO2 == 'x') | (dfO2 == 'X')] = 0
            dfO3[(dfO2 != 'x') & (dfO2 != 'X') & (dfO2.notna())] = 1
            dfO2 = dfO3
        codeLst_cd = [code + '_cd' for code in codeLst]
        return (dfO1.reindex(columns=codeLst), dfO2.reindex(columns=codeLst_cd))
    else:
        return dfO1.reindex(columns=codeLst)


def readStreamflow(siteNo, startDate=None, csv=True):
    """read USGS streamflow (00060) data, did:
    1. fill missing average observation (00060_00003) by available max and min.
    Arguments:
        siteNo {str} -- site number
    Keyword Arguments:
        startDate {date} -- start date (default: {None})
    Returns:
        pandas.DataFrame -- [description]
    """
    if csv is False:
        fileQ = os.path.join(kPath.dirData, 'USGS', 'streamflow', siteNo)
        dfQ = readUsgsText(fileQ, dataType='streamflow')
        if dfQ is None:
            return None
        if startDate is not None:
            dfQ = dfQ[dfQ['date'] >= startDate]
        if '00060_00001' in dfQ.columns and '00060_00002' in dfQ.columns:
            # fill nan using other two fields
            avgQ = dfQ[['00060_00001', '00060_00002']].mean(
                axis=1, skipna=False)
            dfQ['00060_00003'] = dfQ['00060_00003'].fillna(avgQ)
            dfQ = dfQ[['date', '00060_00003']]
        else:
            dfQ = dfQ[['date', '00060_00003']]
    else:
        fileQ = os.path.join(kPath.dirData, 'USGS',
                             'streamflow', 'csv', siteNo)
        dfQ = pd.read_csv(fileQ)
        dfQ['date'] = pd.to_datetime(dfQ['date'], format='%Y-%m-%d')
        if startDate is not None:
            dfQ = dfQ[dfQ['date'] >= startDate]
    return dfQ.set_index('date')


def readUsgsText(fileName, dataType=None):
    """read usgs text file, rename head for given dataType
    Arguments:
        fileName {str} -- file name
    Keyword Arguments:
        dataType {str} -- dailyTS, streamflow or sample (default: {None})
    """
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
        out = renameDailyTS(pdf)
    elif dataType == 'sample':
        out = renameSample(pdf)
    elif dataType == 'streamflow':
        out = renameStreamflow(pdf)
    else:
        out = pdf
    return out


def renameDailyTS(pdf):
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


def renameStreamflow(pdf):
    # pick the longest average Q field
    headLst = pdf.columns.tolist()
    tempS = [head.split('_') for head in headLst if head[-1].isdigit()]
    codeLst = list(set([int(s[0])-int(s[2]) for s in tempS]))
    tempN = list()
    for code in codeLst:
        for k in range(3):
            head = '{}_00060_{:05n}'.format(code+k+1, k+1)
            if head not in headLst:
                pdf[head] = np.nan
                pdf[head+'_cd'] = 'N'
        tempLst = ['{}_00060_{:05n}'.format(code+k+1, k+1) for k in range(3)]
        temp = ((~pdf[tempLst[0]].isna()) & (~pdf[tempLst[1]].isna())) | (
            ~pdf[tempLst[2]].isna())
        tempN.append(temp.sum())
    code = codeLst[tempN.index(max(tempN))]
    # (searched and no code of leading zero)
    pdf = pdf.rename(columns={'{}_00060_{:05n}'.format(
        code+x+1, x+1): '00060_{:05n}'.format(x+1) for x in range(3)})
    pdf = pdf.rename(columns={'{}_00060_{:05n}_cd'.format(
        code+x+1, x+1): '00060_{:05n}_cd'.format(x+1) for x in range(3)})

    # time field
    pdf['date'] = pd.to_datetime(pdf['datetime'], format='%Y-%m-%d')
    return pdf


def renameSample(pdf):
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


def removeFlag(dfC, dfCF):
    codeLstF = dfCF.columns.tolist()
    codeLst = [code[:5] for code in codeLstF]
    dfOut = dfC.copy()
    data = dfC[codeLst].values
    dataF = dfCF[codeLstF].values
    data[dataF == 1] = np.nan
    dfOut[codeLst] = data
    return dfOut
