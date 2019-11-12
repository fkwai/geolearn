import numpy as np
import scipy
import time
import statsmodels.api as sm


class statError(object):
    def __init__(self, *, pred, target):
        ngrid, nt = pred.shape
        # Bias
        self.Bias = np.nanmean(pred - target, axis=1)
        # RMSE
        self.RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
        # ubRMSE
        predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
        targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
        predAnom = pred - predMean
        targetAnom = target - targetMean
        self.ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
        # rho
        rho = np.full(ngrid, np.nan)
        for k in range(0, ngrid):
            x = pred[k, :]
            y = target[k, :]
            ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
            if ind.shape[0] > 0:
                xx = x[ind]
                yy = y[ind]
                rho[k] = scipy.stats.pearsonr(xx, yy)[0]
        self.rho = rho


class statSigma(object):
    def __init__(self, *, dataMC, dataSigma, dataSigmaBatch):
        if dataMC is not None:
            self.sigmaMC_mat = np.std(dataMC, axis=2)
            self.sigmaMC = np.sqrt(np.nanmean(self.sigmaMC_mat**2, axis=1))
        if dataSigma is not None:
            self.sigmaX_mat = dataSigma
            self.sigmaX = np.sqrt(np.nanmean(self.sigmaX_mat**2, axis=1))
        if dataMC is not None and dataSigma is not None:
            self.sigma_mat = np.sqrt(self.sigmaMC_mat**2 + self.sigmaX_mat**2)
            self.sigma = np.sqrt(np.mean(self.sigma_mat**2, axis=1))
        # if dataSigmaBatch is not None:
        #     self.sigma_mat = np.sqrt(np.nanmean(dataSigmaBatch**2, axis=2))
        #     self.sigma = np.sqrt(np.nanmean(self.sigma_mat**2, axis=1))

    def regComb(self, dsReg, field='LSTM', opt=1, fTest=None):
        statSigma = dsReg.statCalSigma(field=field)
        # do regression
        if opt == 1:
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = statSigma.sigmaMC_mat * statSigma.sigmaX_mat
            y = np.square(dsReg.LSTM-dsReg.SMAP) - \
                np.square(statSigma.sigmaX_mat)
            xx = np.stack((x1.flatten(), x2.flatten()), axis=1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 2:
            x1 = np.square(statSigma.sigmaMC_mat)
            y = np.square(dsReg.LSTM-dsReg.SMAP) - \
                np.square(statSigma.sigmaX_mat)
            xx = x1.flatten().reshape(-1, 1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 3:
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = np.square(statSigma.sigmaX_mat)
            x3 = statSigma.sigmaMC_mat * statSigma.sigmaX_mat
            x4 = np.ones(x1.shape)
            y = np.square(dsReg.LSTM - dsReg.SMAP)
            xx = np.stack(
                (x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()),
                axis=1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 4:
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = np.square(statSigma.sigmaX_mat)
            x3 = np.ones(x1.shape)
            y = np.square(dsReg.LSTM - dsReg.SMAP)
            xx = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 5:
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = np.square(statSigma.sigmaX_mat)
            x3 = statSigma.sigmaMC_mat * statSigma.sigmaX_mat
            y = np.square(dsReg.LSTM - dsReg.SMAP)
            xx = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 6:
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = np.square(statSigma.sigmaX_mat)
            y = np.square(dsReg.LSTM - dsReg.SMAP)
            xx = np.stack((x1.flatten(), x2.flatten()), axis=1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 7:
            x1 = np.square(statSigma.sigmaMC_mat)
            y = np.square(dsReg.LSTM - dsReg.SMAP)
            xx = x1.flatten().reshape(-1, 1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 8:
            x1 = np.square(statSigma.sigmaX_mat)
            y = np.square(dsReg.LSTM - dsReg.SMAP)
            xx = x1.flatten().reshape(-1, 1)
            yy = y.flatten().reshape(-1, 1)
        elif opt == 9:
            x1 = np.ones(statSigma.sigma_mat.shape)
            y = np.square(dsReg.LSTM-dsReg.SMAP) - \
                np.square(statSigma.sigma_mat)
            xx = x1.flatten().reshape(-1, 1)
            yy = y.flatten().reshape(-1, 1)

        ind = np.where(~np.isnan(yy))[0]
        xf = xx[ind, :]
        yf = yy[ind]
        # w, _, _, _ = np.linalg.lstsq(xf, yf)
        # model = sm.OLS(yf, xf)
        model = sm.RLM(yf, xf)
        result = model.fit()
        w = result.params
        if fTest is not None:
            ftestP = list()
            ftestF = list()
            for k in range(len(w)):
                ww = w.copy()
                ww[k] = fTest[k]
                ff = result.f_test(ww)
                ftestP.append(ff.pvalue)
                ftestF.append(ff.fvalue)

        if opt == 1:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                self.sigmaMC_mat * self.sigmaX_mat * w[1] +
                np.square(self.sigmaX_mat))
            k = -w[1] / 2
            a = w[0] - k**2
            out = [a, k]
        elif opt == 2:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat))
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = np.ones(x1.shape)
            y = np.square(statSigma.sigmaX_mat)
            xx = np.stack((x1.flatten(), x2.flatten()), axis=1)
            yy = y.flatten().reshape(-1, 1)
            k, _, _, _ = np.linalg.lstsq(xx, yy)
            k = k[0]
            a = w[0] + k
            out = [a, k]
        elif opt == 3:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1] +
                self.sigmaMC_mat * self.sigmaX_mat * w[2] +
                np.ones(self.sigmaX_mat.shape) * w[3])
            out = w
        elif opt == 4:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1] +
                np.ones(self.sigmaX_mat.shape) * w[2])
            out = w
        elif opt == 5:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1] +
                self.sigmaMC_mat * self.sigmaX_mat * w[2])
            out = w
        elif opt == 6:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1])
            out = w
        elif opt == 7:
            self.sigmaReg_mat = np.sqrt(np.square(self.sigmaMC_mat) * w[0])
            out = w
        elif opt == 8:
            self.sigmaReg_mat = np.sqrt(np.square(self.sigmaX_mat) * w[0])
            out = w
        elif opt == 9:
            self.sigmaReg_mat = np.sqrt(np.square(self.sigma_mat) + w[0])
            out = w
        self.sigmaReg = np.sqrt(np.mean(self.sigmaReg_mat**2, axis=1))
        if fTest is None:
            return result
        else:
            return (out, ftestP, ftestF)


    def regComb2(self,
                dsReg,
                predField='LSTM',
                targetField='SMAP',
                opt=1,
                fTest=None):
        statSigma = dsReg.statCalSigma(field=predField)
        statErr = dsReg.statCalError(predField=predField, targetField=targetField)
        y = np.square(statErr.ubRMSE)

        # do regression
        if opt == 1:
            x1 = np.square(statSigma.sigmaMC)
            x2 = statSigma.sigmaMC * statSigma.sigmaX
            xx = np.stack((x1, x2), axis=1)
            yy = y - np.square(statSigma.sigmaX)
        elif opt == 2:
            x1 = np.square(statSigma.sigmaMC)
            xx = x1.reshape(-1, 1)
            yy = y - np.square(statSigma.sigmaX)
        elif opt == 3:
            x1 = np.square(statSigma.sigmaMC)
            x2 = np.square(statSigma.sigmaX)
            x3 = statSigma.sigmaMC * statSigma.sigmaX
            x4 = np.ones(x1.shape)
            xx = np.stack((x1, x2, x3, x4), axis=1)
            yy = y
        elif opt == 4:
            x1 = np.square(statSigma.sigmaMC)
            x2 = np.square(statSigma.sigmaX)
            x3 = np.ones(x1.shape)
            xx = np.stack((x1, x2, x3), axis=1)
            yy = y
        elif opt == 5:
            x1 = np.square(statSigma.sigmaMC)
            x2 = np.square(statSigma.sigmaX)
            x3 = statSigma.sigmaMC * statSigma.sigmaX
            xx = np.stack((x1, x2, x3), axis=1)
            yy = y
        elif opt == 6:
            x1 = np.square(statSigma.sigmaMC)
            x2 = np.square(statSigma.sigmaX)
            xx = np.stack((x1, x2), axis=1)
            yy = y
        elif opt == 7:
            x1 = np.square(statSigma.sigmaMC)
            xx = x1.reshape(-1, 1)
            yy = y
        elif opt == 8:
            x1 = np.square(statSigma.sigmaX)
            xx = x1.reshape(-1, 1)
            yy = y
        elif opt == 9:
            x1 = np.ones(statSigma.sigma.shape)
            xx = x1.reshape(-1, 1)
            yy = y - np.square(statSigma.sigma)

        ind = np.where(~np.isnan(yy))[0]
        xf = xx[ind, :]
        yf = yy[ind]
        # w, _, _, _ = np.linalg.lstsq(xf, yf)
        # model = sm.OLS(yf, xf)
        model = sm.RLM(yf, xf)
        result = model.fit()
        w = result.params
        if fTest is not None:
            ftestP = list()
            ftestF = list()
            for k in range(len(w)):
                ww = w.copy()
                ww[k] = fTest[k]
                ff = result.f_test(ww)
                ftestP.append(ff.pvalue)
                ftestF.append(ff.fvalue)

        if opt == 1:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                self.sigmaMC_mat * self.sigmaX_mat * w[1] +
                np.square(self.sigmaX_mat))
            k = -w[1] / 2
            a = w[0] - k**2
            out = [a, k]
        elif opt == 2:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] + np.square(self.sigmaX_mat))
            x1 = np.square(statSigma.sigmaMC_mat)
            x2 = np.ones(x1.shape)
            y = np.square(statSigma.sigmaX_mat)
            xx = np.stack((x1.flatten(), x2.flatten()), axis=1)
            yy = y.flatten().reshape(-1, 1)
            k, _, _, _ = np.linalg.lstsq(xx, yy)
            k = k[0]
            a = w[0] + k
            out = [a, k]
        elif opt == 3:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1] +
                self.sigmaMC_mat * self.sigmaX_mat * w[2] +
                np.ones(self.sigmaX_mat.shape) * w[3])
            out = w
        elif opt == 4:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1] +
                np.ones(self.sigmaX_mat.shape) * w[2])
            out = w
        elif opt == 5:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1] +
                self.sigmaMC_mat * self.sigmaX_mat * w[2])
            out = w
        elif opt == 6:
            self.sigmaReg_mat = np.sqrt(
                np.square(self.sigmaMC_mat) * w[0] +
                np.square(self.sigmaX_mat) * w[1])
            out = w
        elif opt == 7:
            self.sigmaReg_mat = np.sqrt(np.square(self.sigmaMC_mat) * w[0])
            out = w
        elif opt == 8:
            self.sigmaReg_mat = np.sqrt(np.square(self.sigmaX_mat) * w[0])
            out = w
        elif opt == 9:
            self.sigmaReg_mat = np.sqrt(np.square(self.sigma_mat) + w[0])
            out = w
        self.sigmaReg = np.sqrt(np.mean(self.sigmaReg_mat**2, axis=1))
        if fTest is None:
            return result
        else:
            return (out, ftestP, ftestF)


class statConf(object):
    def __init__(self,
                 *,
                 statSigma,
                 dataPred,
                 dataTarget,
                 dataMC,
                 rmBias=False):
        u = dataPred
        y = dataTarget
        if rmBias is True:
            [ng, nt] = u.shape
            b = np.nanmean(u, axis=1) - np.nanmean(y, axis=1)
            u = u - np.tile(b[:, None], [1, nt])
        # z = np.nanmean(dataMC, axis=2)
        # sigmaLst = ['sigmaMC', 'sigmaX', 'sigma']

        if hasattr(statSigma, 'sigmaX_mat'):
            s = getattr(statSigma, 'sigmaX_mat')
            conf = scipy.special.erf(-np.abs(y - u) / s / np.sqrt(2)) + 1
            setattr(self, 'conf_sigmaX', conf)

        if hasattr(statSigma, 'sigma_mat'):
            s = getattr(statSigma, 'sigma_mat')
            conf = scipy.special.erf(-np.abs(y - u) / s / np.sqrt(2)) + 1
            setattr(self, 'conf_sigma', conf)

        if hasattr(statSigma, 'sigmaReg_mat'):
            s = getattr(statSigma, 'sigmaReg_mat')
            conf = scipy.special.erf(-np.abs(y - u) / s / np.sqrt(2)) + 1
            setattr(self, 'conf_sigmaReg', conf)

        # if hasattr(statSigma, 'sigmaComb_mat'):
        #     s = getattr(statSigma, 'sigmaComb_mat')
        #     conf = scipy.special.erf(np.abs(y-u)/s/np.sqrt(2))
        #     setattr(self, 'conf_sigmaComb', conf)

        if hasattr(statSigma, 'sigmaMC_mat'):
            n = dataMC.shape[2]
            # yR = np.stack((y, 2*u-y), axis=2)
            # yRsort = np.sort(yR, axis=2)
            # bMat1 = dataMC <= np.tile(yRsort[:, :, 0:1], [1, 1, n])
            # n1 = np.count_nonzero(bMat1, axis=2)
            # bMat2 = dataMC >= np.tile(yRsort[:, :, 1:2], [1, 1, n])
            # n2 = np.count_nonzero(bMat2, axis=2)
            # conf = (n1+n2)/n
            # conf[np.isnan(y)] = np.nan

            # y = dataMC[:, :, 0]
            dmat = np.tile(np.abs(y - u)[:, :, None], [1, 1, n])
            dmatMC = np.abs(dataMC - np.tile(u[:, :, None], [1, 1, n]))
            bMat = dmatMC >= dmat
            n1 = np.count_nonzero(bMat, axis=2)
            conf = n1 / n
            conf[np.isnan(y)] = np.nan

            # m1 = np.concatenate((y[:, :, None], dataMC), axis=2)
            # m2 = np.concatenate(((2*u-y)[:, :, None], dataMC), axis=2)
            # # rm1 = np.argsort(m1)[:, :, 0]
            # rm1 = np.where(np.argsort(m1) == 0)[2].reshape(y.shape[0],y.shape[1])
            # # rm2 = np.argsort(m2)[:, :, 0]
            # rm2 = np.where(np.argsort(m2) == 0)[2].reshape(y.shape[0], y.shape[1])
            # conf = 1-np.abs(rm1-rm2)/n
            # conf[np.isnan(y)] = np.nan
            setattr(self, 'conf_sigmaMC', conf)


class statProb(object):
    def __init__(self, *, statSigma, dataPred, dataTarget, dataMC):
        u = dataPred
        y = dataTarget
        z = np.nanmean(dataMC, axis=2)
        # sigmaLst = ['sigmaMC', 'sigmaX', 'sigma']

        if hasattr(statSigma, 'sigmaX_mat'):
            s = getattr(statSigma, 'sigmaX_mat')
            # prob = scipy.special.erf(np.abs(y-u)/s/np.sqrt(2))
            prob = scipy.stats.norm.pdf((y - u) / s)
            setattr(self, 'prob_sigmaX', prob)

        if hasattr(statSigma, 'sigma_mat'):
            s = getattr(statSigma, 'sigma_mat')
            # prob = scipy.special.erf(np.abs(y-z)/s/np.sqrt(2))
            prob = scipy.stats.norm.pdf((y - u) / s)
            setattr(self, 'prob_sigma', prob)

        if hasattr(statSigma, 'sigmaComb_mat'):
            s = getattr(statSigma, 'sigmaComb_mat')
            # prob = scipy.special.erf(np.abs(y-u)/s/np.sqrt(2))
            prob = scipy.stats.norm.pdf((y - u) / s)
            setattr(self, 'prob_sigmaComb', prob)

        if hasattr(statSigma, 'sigmaMC_mat'):
            n = dataMC.shape[2]
            m = np.concatenate((y[:, :, None], dataMC), axis=2)
            rm = np.argsort(m)[:, :, 0]
            prob = 1 - np.abs(2 * rm - n) / n
            prob[np.isnan(y)] = np.nan
            setattr(self, 'prob_sigmaMC', prob)


class statNorm(object):
    def __init__(self, *, statSigma, dataPred, dataTarget):
        u = dataPred
        y = dataTarget
        sigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
        for sigmaStr in sigmaLst:
            if hasattr(statSigma, sigmaStr + '_mat'):
                s = getattr(statSigma, sigmaStr + '_mat')
                yNorm = (y - u) / s
                setattr(self, 'yNorm_' + sigmaStr, yNorm)
