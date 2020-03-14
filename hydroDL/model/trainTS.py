import numpy as np
import torch
import time
from hydroDL.model import rnn, crit
from hydroDL.data import transform


def subsetRandom(dataLst, batchSize, sizeLst=None):
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
    [nbatch, rho] = batchSize
    [nx, nxc, ny, nyc, ns, nt] = sizeLst

    iR = np.random.randint(0, ns, [nbatch])
    xTemp = x[nt-rho:rho, iR, :]
    xcTemp = np.tile(xc[iR, :], [nt, 1, 1])
    xTensor = torch.from_numpy(np.concatenate(
        [xTemp, xcTemp], axis=-1)).float()
    yTemp = y[nt-rho:rho, iR, :]
    ycTemp = np.full([rho, nbatch, nyc], np.nan)
    ycTemp[-1, :, :] = yc[iR, :]
    yTensor = torch.from_numpy(np.concatenate(
        [yTemp, ycTemp], axis=-1)).float()
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def trainModel(dataLst, model, lossFun, optim, batchSize=[100, 365], nEp=100, cEp=0):
    """[summary]    
    Arguments:
        dataLst {list} --  see trainModel [x,xc,y,yc]
            x {np.array} -- input time series of size [nt,np,nx]
            xc {np.array} -- input constant of size [np,nxc]
            y {np.array} -- target time series of size [nt,np,ny]
            yc {np.array} -- target constant (or last time step) of size [np,nyc]
        batchSize {list} -- [spatial batch size, temporal batch size]
        model {[type]} -- [description]
        lossFun {[type]} -- [description]

    Keyword Arguments:
        batchSize {list} -- [description] (default: {[100, 365]})
        nEp {int} -- [number of epochs to run] (default: {100})
        cEp {int} -- [current epoch (only for print)] (default: {0})

    Returns:
        [type] -- [description]
    """
    x, xc, y, yc = dataLst
    nbatch, rho = batchSize
    nt, ns, nx = x.shape
    nt, ns, ny = y.shape
    ns, nxc = xc.shape
    ns, nyc = yc.shape
    sizeLst = [nx, nxc, ny, nyc, ns, nt]

    # training
    if nbatch > ns:
        nIterEp = 1
    else:
        nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - nbatch / ns)))

    lossEp = 0
    lossEpLst = list()
    t0 = time.time()
    model.train()
    model.zero_grad()
    for iEp in range(1, nEp + 1):
        lossEp = 0
        t0 = time.time()
        # somehow the first iteration always failed
        try:
            xT, yT = subsetRandom(dataLst, batchSize, sizeLst)
            yP = model(xT)
        except:
            print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')

        for iIter in range(nIterEp):
            xT, yT = subsetRandom(dataLst, batchSize, sizeLst)
            try:
                yP = model(xT)
                loss = lossFun(yP, yT)
                loss.backward()
                optim.step()
                model.zero_grad()
                lossEp = lossEp + loss.item()
            except:
                print('iteration Failed: iter {} ep {}'.format(iIter, iEp+cEp))
        lossEp = lossEp / nIterEp
        ct = time.time() - t0
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, lossEp, ct)
        print(logStr)
        lossEpLst.append(lossEp)
    return model, optim, lossEpLst


def testModel(model, x, xc, batchSize=100):
    nt, ns, nx = x.shape
    ns, nxc = xc.shape
    iS = np.arange(0, ns, batchSize)
    iE = np.append(iS[1:], ns)
    yOutLst = list()
    for k in range(len(iS)):
        print('batch: '+str(k))
        xT = torch.from_numpy(np.concatenate(
            [x[:, iS[k]:iE[k], :], np.tile(xc[iS[k]:iE[k], :], [nt, 1, 1])], axis=-1)).float()
        if torch.cuda.is_available():
            xT = xT.cuda()
            model = model.cuda()
        yT = model(xT)
        yOutLst.append(yT.detach().cpu().numpy())
    yOut = np.concatenate(yOutLst, axis=1)
    return yOut


def saveModel(model, modelFile):
    torch.save(model, modelFile)
