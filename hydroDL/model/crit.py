import torch


def nanmean(x, *args, **kwargs):
    # could be replaced in newer torch version
    v = x.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == RmseLoss
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class RmseEnd(torch.nn.Module):
    def __init__(self):
        super(RmseEnd, self).__init__()

    def forward(self, output, target):
        mask = ~torch.isnan(target[-1, :, :])
        p = output[-1, :, :][mask]
        t = target[-1, :, :][mask]
        loss = torch.sqrt(((p - t)**2).mean())
        return loss


class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        nt, ns, ny = target.shape
        loss = 0
        for k in range(ny):
            n = 0
            lossTemp = 0
            for j in range(ns):
                p0 = output[:, j, k]
                t0 = target[:, j, k]
                mask = t0 == t0
                if len(mask[mask == True]) > 0:
                    p = p0[mask]
                    t = t0[mask]
                    temp = torch.sqrt(((p - t)**2).mean())
                    lossTemp = lossTemp + temp
                    n = n+1
            loss = loss + lossTemp/n
        return loss/ny


class RmseSample(torch.nn.Module):
    def __init__(self):
        super(RmseSample, self).__init__()

    def forward(self, output, target):
        D = (output-target)**2
        nt, ns, ny = target.shape
        loss = 0
        for k in range(ny):
            tempLoss = 0
            cs = 0
            for j in range(ns):
                d = D[:, j, k]
                mask = d == d
                l = d[mask].mean()
                if l == l:
                    tempLoss = tempLoss+l
                    cs = cs+1
            if cs > 0:
                loss = loss + tempLoss/cs
        return loss/ny


class RmseLoss2D(torch.nn.Module):
    def __init__(self):
        super(RmseLoss2D, self).__init__()

    def forward(self, output, target):
        ny = target.shape[1]
        loss = 0
        for k in range(ny):
            p0 = output[:,  k]
            t0 = target[:,  k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            if temp == temp:
                loss = loss + temp
        return loss/ny


class RmseLoss3D(torch.nn.Module):
    def __init__(self):
        super(RmseLoss3D, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            if temp == temp:
                loss = loss + temp
        return loss/ny


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = ((p - t)**2).mean()
            loss = loss + temp
        return loss


class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            losssum = 0
            nsample = 0
            for ii in range(Ngage):
                p0 = output[:, ii, k]
                t0 = target[:, ii, k]
                mask = t0 == t0
                if torch.any(mask):
                    p = p0[mask]
                    t = t0[mask]
                    SST = torch.sum((t - t.mean()) ** 2)
                    if SST != 0:
                        SSRes = torch.sum((t - p) ** 2)
                        temp = 1 - SSRes / SST
                        losssum = losssum + temp
                        nsample = nsample + 1
            loss = -(losssum/nsample)+loss
        return loss


class NSELoss2D(torch.nn.Module):
    def __init__(self):
        super(NSELoss2D, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii]
            t0 = target[:, ii]
            mask = t0 == t0
            if torch.any(mask):
                p = p0[mask]
                t = t0[mask]
                SST = torch.sum((t - t.mean()) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample + 1
        return losssum/nsample


class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask == True]) > 0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                temp = SSRes / ((torch.sqrt(SST)+0.1)**2)
                losssum = losssum + temp
                nsample = nsample + 1
        loss = losssum/nsample
        return loss


class LogLoss2D(torch.nn.Module):
    def __init__(self):
        super(LogLoss2D, self).__init__()

    def forward(self, pred, targ):
        ns = targ.shape[1]
        lossTemp = 0
        n = 0
        for k in range(ns):
            iv = ~torch.isnan(targ[:, k])
            if len(iv[iv == True]) > 10:
                rmse = torch.mean((pred[iv, k]-targ[iv, k])**2, dim=0)
                lossTemp = lossTemp+torch.log(rmse+1e-8)
                n = n+1
        if n == 0:
            return 0
        else:
            return torch.exp(lossTemp/n)
        # return mse


class LogLoss3D(torch.nn.Module):
    def __init__(self):
        super(LogLoss3D, self).__init__()

    def forward(self, pred, targ):
        nt, ns, ny = targ.shape
        loss2 = 0.0
        n2 = 0
        for i in range(ny):
            loss1 = 0
            n1 = 0
            for j in range(ns):
                iv = ~torch.isnan(targ[:, j, i])
                if iv.sum() > 5:
                    mse = torch.mean(
                        (pred[iv, j, i]-targ[iv, j, i])**2, dim=0)
                    loss1 = loss1+torch.log(mse+1e-8)
                    n1 = n1+1
            if n1 > 0:
                loss2 = loss2+loss1/n1
                n2 = n2+1
        if n2 == 0:
            return torch.tensor([0.], requires_grad=True)
        else:
            return torch.exp(loss2/n2)


class NashLoss2D(torch.nn.Module):
    def __init__(self):
        super(NashLoss2D, self).__init__()

    def forward(self, pred, targ):
        loss = 0
        nt, ns = pred.shape
        n = 0
        for iS in range(ns):
            p = pred[:, iS]
            t = targ[:, iS]
            mask = ~torch.isnan(t)
            if mask.sum() > 10:
                pp = p[mask]
                tt = t[mask]
                sst = torch.sum((tt-tt.mean())**2)
                res = torch.sum((tt-pp)**2)
                if sst != 0:
                    loss = loss + res/(torch.sqrt(sst)+0.1)**2
                    n = n+1
        return loss/n


class NashLoss3D(torch.nn.Module):
    def __init__(self):
        super(NashLoss3D, self).__init__()

    def forward(self, pred, targ):
        loss = 0
        nt, ns, ny = pred.shape
        for iY in range(ny):
            lossY = 0
            n = 0
            for iS in range(ns):
                p = pred[:, iS, iY]
                t = targ[:, iS, iY]
                mask = ~torch.isnan(t)
                if mask.sum() > 10:
                    pp = p[mask]
                    tt = t[mask]
                    sst = torch.sum((tt-tt.mean())**2)
                    res = torch.sum((tt-pp)**2)
                    if sst != 0:
                        lossY = lossY + res/(torch.sqrt(sst)+0.1)**2
                        n = n+1
            if n > 0:
                loss = loss+lossY/n
        return loss/ny


# class LogLoss2D(torch.nn.Module):
#     def __init__(self):
#         super(LogLoss2D, self).__init__()

#     def forward(self, pred, targ):
#         ns = targ.shape[1]
#         iv = ~torch.isnan(targ)
#         mse = torch.nansum((pred[iv]-targ[iv])**2, dim=0)
#         n = torch.sum(~torch.isnan(targ), dim=0)
#         return torch.exp(torch.sum(torch.log(mse/n))/ns)


class LogAll2D(torch.nn.Module):
    def __init__(self):
        super(LogAll2D, self).__init__()

    def forward(self, pred, targ):
        sn = 1e-8
        ns = targ.shape[1]
        logD = torch.log(torch.abs(pred-targ)+sn)
        logLoss = torch.nansum(logD, dim=0) / \
            torch.sum(~torch.isnan(targ), dim=0)
        loss = torch.nansum(logLoss)/ns
        return loss
