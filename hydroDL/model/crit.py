import torch


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


class logLoss2D(torch.nn.Module):
    def __init__(self):
        super(logLoss2D, self).__init__()

    def forward(self, pred, targ):
        ns = targ.shape[1]
        mse = torch.nansum((pred-targ)**2, dim=0)
        n = torch.sum(~torch.isnan(targ), dim=0)
        return torch.exp(torch.sum(torch.log(mse/n))/ns)


class LogLoss2D(torch.nn.Module):
    def __init__(self):
        super(LogLoss2D, self).__init__()

    def forward(self, pred, targ):
        ns = targ.shape[1]
        mse = torch.nansum((pred-targ)**2, dim=0)
        n = torch.sum(~torch.isnan(targ), dim=0)
        return torch.exp(torch.sum(torch.log(mse/n))/ns)


class LogAll2D(torch.nn.Module):
    def __init__(self):
        super(LogAll2D, self).__init__()

    def forward(self, pred, targ):
        sn = 1e-8
        ns = targ.shape[1]
        logD = torch.log(torch.abs(pred-targ)+sn)
        logLoss = torch.nansum(logD, dim=0) / \
            torch.sum(~torch.isnan(targ), dim=0)
        loss = torch.exp(torch.nansum(logLoss)/ns)
        return loss
