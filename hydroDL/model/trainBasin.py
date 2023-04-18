import numpy as np
import torch
import time
from hydroDL.model import rnn, crit
from hydroDL.data import transform
import pickle
from datetime import datetime
import os


def subsetRandom(dataLst, batchSize, sizeLst, opt='Random', wS=None, wT=None):
    """get a random subset of training data
    Arguments:
        dataLst {list} --  see trainModel [x,xc,y,yc]
        batchSize {list} -- [spatial batch size, temporal batch size]
        sizeLst {list} -- list of following: (default: {None})
            nx {int} -- # input time series (default: {None})
            nxc {int} -- # input constant (default: {None})
            ny {int} -- # target time series (default: {None})
            nyc {int} -- # target constant (default: {None})
            ns {int} -- # pixels / instances (default: {None})
            nt {int} -- # time steps (default: {None})
    Returns:
        [torch.Tensor torch.Tensor] -- training subset
    """
    [x, xc, y, yc] = dataLst
    [rho, nbatch] = batchSize
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
    if opt == 'Index':
        iS = np.random.randint(0, ns, [nbatch])
        xTemp = x[:, iS, :]
        yTemp = y[:, iS, :] if y is not None else np.ndarray([nt, nbatch, 0])
    else:
        if opt == 'Random':
            iS = np.random.randint(0, ns, [nbatch])
            iT = np.random.randint(0, nt - rho + 1, [nbatch])
        elif opt == 'Weight':
            iS = np.random.choice(ns, nbatch, p=wS)
            iT = np.zeros(nbatch).astype(int)
            for k in range(nbatch):
                iT[k] = np.random.choice(nt - rho + 1, p=wT[:, iS[k]])
        else:
            raise ValueError('opt must be Index, Random or Weight')
        xTemp = np.full([rho, nbatch, nx], np.nan)        
        yTemp = np.full([rho, nbatch, ny], np.nan)        
        if x is not None:
            for k in range(nbatch):
                xTemp[:, k, :] = x[iT[k] : iT[k] + rho, iS[k], :]
        if y is not None:
            for k in range(nbatch):
                yTemp[:, k, :] = y[iT[k] : iT[k] + rho, iS[k], :]
    xcTemp = np.full([rho, nbatch, nxc], np.nan)
    ycTemp = np.full([rho, nbatch, nyc], np.nan)

    if xc is not None:
        xcTemp = np.tile(xc[iS, :], [rho, 1, 1])
    if yc is not None:
        ycTemp[-1, :, :] = yc[iS, :]
    xTensor = torch.from_numpy(np.concatenate([xTemp, xcTemp], axis=-1)).float()
    yTensor = torch.from_numpy(np.concatenate([yTemp, ycTemp], axis=-1)).float()
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def getSize(dataTup):
    # dataTup: [x,xc,y,yc]
    sizeLst = list()
    for data in dataTup:
        if data is None:
            sizeLst.append(0)
        else:
            sizeLst.append(data.shape[-1])
    [nx, nxc, ny, nyc] = sizeLst
    x = dataTup[0]
    nt = x.shape[0]
    ns = x.shape[1]
    return (nx, nxc, ny, nyc, nt, ns)


def dealNaN(dataTup, optNaN):
    # check if any nan
    rmLst = list()
    dataLst = list()
    for data, optN in zip(dataTup, optNaN):
        if data is not None:
            data[np.isinf(data)] = np.nan
            indNan = np.where(np.isnan(data))
            if len(indNan[0]) > 0:
                if optN == 1:
                    data[indNan] = -1
                    print('nan found and filled -1')
                elif optN == 2:
                    if data.ndim == 2:
                        rmLst.append(indNan[0])
                    if data.ndim == 3:
                        rmLst.append(np.unique(np.where(np.isnan(data))[1]))
        dataLst.append(data)
    if len(rmLst) > 0:
        rmAry = np.concatenate(rmLst)
        for k in range(len(dataLst)):
            if dataLst[k] is not None:
                dataLst[k] = np.delete(dataLst[k], rmAry, axis=dataLst[k].ndim - 2)
        print('nan found and removed')
    return dataLst


def batchWeight(matY, rho, nbatch, opt='obsDay'):
    matB = ~np.isnan(matY)
    # calculate batch weights based on Y observations
    if opt == 'obsDay':
        matD = np.any(matB, axis=2)
        filt = np.ones(rho, dtype=int)
        countT = np.apply_along_axis(
            lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=matD
        )
        nT = np.sum(countT, axis=0)
        wS = nT / np.sum(countT)
        wT = countT / nT
        wT[np.isnan(wT)] = 0
        pr = np.mean(countT) * nbatch / np.sum(matD)
    return pr, wS, wT


def trainModel(
    dataLst,
    model,
    lossFun,
    optim,
    batchSize=[None, 100],
    nEp=100,
    cEp=0,
    optBatch='Random',
    nIterEp=None,
    outFolder=None,
    logH=None,
):
    """[summary]
    Arguments:
        dataLst {list} --  see trainModel [x,xc,y,yc]
            x {np.array} -- input time series of size [nt,np,nx]
            xc {np.array} -- input constant of size [np,nxc]
            y {np.array} -- target time series of size [nt,np,ny]
            yc {np.array} -- target constant (or last time step) of size [np,nyc]
        batchSize {list} -- [spatial batch size, temporal batch size, None for use all]
        model {[type]} -- [description]
        lossFun {[type]} -- [description]

    Keyword Arguments:
        batchSize {list} -- [description] (default: {[100, 365]})
        nEp {int} -- [number of epochs to run] (default: {100})
        cEp {int} -- [current epoch (only for print)] (default: {0})
        optBatch {str} -- [description] (default: {'Random'})
        nIterEp {int} -- [number of iterations per epoch] (default: {None})
        outFolder {str} -- [in case of save debug files] (default: {None})

    Returns:
        [type] -- [description]
    """
    sizeLst = getSize(dataLst)
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
    rho, nbatch = batchSize
    if rho is None:
        rho = nt
    if nbatch is None:
        nbatch = ns
    batchSize = [rho, nbatch]

    # training
    if optBatch == 'Random':
        pr = nbatch * rho / ns / nt
        wS = None
        wT = None
    elif optBatch == 'Weight':
        pr, wS, wT = batchWeight(dataLst[2], rho, nbatch)
    elif optBatch == 'Index':
        pr = nbatch / ns
        wS, wT = None, None
    if nIterEp is None:
        nIterEp = 1 if pr > 1 else int(np.ceil(np.log(0.01) / np.log(1 - pr)))
    print('iter per epoch {}'.format(nIterEp), flush=True)
    t0 = time.time()
    model.train()
    for iEp in range(1, nEp + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(nIterEp):
            xT, yT = subsetRandom(
                dataLst, batchSize, sizeLst, opt=optBatch, wS=wS, wT=wT
            )
            model.zero_grad(set_to_none=True)
            yP = model(xT)
            if type(lossFun) is crit.RmseLoss2D:
                loss = lossFun(yP[-1, :, :], yT[-1, :, :])
            else:
                loss = lossFun(yP, yT)
            loss.backward()
            # skip and debug nan grad
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    strTime = datetime.now().strftime('%m%d%H%M%S')
                    fileName = 'debug{}-{}-{}'.format(iEp, iIter, strTime)
                    if outFolder is None:
                        filePath = fileName
                    else:
                        filePath = os.path.join(outFolder, fileName)
                    print('nan in grad, skipped and see {}'.format(fileName))
                    with open(filePath, 'wb') as handle:
                        pickle.dump(
                            dict(xT=xT, yT=yT, lossFun=lossFun, model=model), handle
                        )
                    model.zero_grad()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            lossEp = lossEp + loss.item()
            # except:
            #     print('iteration Failed: iter {} ep {}'.format(iIter, iEp+cEp))
        lossEp = lossEp / nIterEp
        ct = time.time() - t0
        logStr = 'Epoch {} Loss {} time {:.2f} '.format(iEp + cEp, lossEp, ct)
        if logH is not None:
            logH.write(logStr + '\n')
            logH.flush()
        print(logStr, flush=True)
    return model, optim


def testModel(model, x, xc, ny=None, batchSize=100):
    model.eval()
    nt, ns, nx = x.shape
    iS = np.arange(0, ns, batchSize)
    iE = np.append(iS[1:], ns)
    yLst = list()
    ycLst = list()
    if batchSize > ns:
        batchSize = ns
    for k in range(len(iS)):
        print('batch: ' + str(k))
        if xc is not None:
            xT = torch.from_numpy(
                np.concatenate(
                    [x[:, iS[k] : iE[k], :], np.tile(xc[iS[k] : iE[k], :], [nt, 1, 1])],
                    axis=-1,
                )
            ).float()
        else:
            xT = torch.from_numpy(x[:, iS[k] : iE[k], :]).float()
        if torch.cuda.is_available():
            xT = xT.cuda()
            model = model.cuda()
        if k == 0:
            try:
                yT = model(xT)
            except:
                print('first iteration failed again')
        yT = model(xT)
        out = yT.detach().cpu().numpy()
        if ny is None:
            yLst.append(out)
        else:
            yLst.append(out[:, :, :ny])
            ycLst.append(out[-1, :, ny:])
    if ny is None:
        yOut = np.concatenate(yLst, axis=1)
        return yOut
    else:
        yOut = np.concatenate(yLst, axis=1)
        ycOut = np.concatenate(ycLst, axis=0)
        return yOut, ycOut


def saveModel(model, modelFile):
    torch.save(model, modelFile)
