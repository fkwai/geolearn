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
            mask = t0 == t0
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
        mask = target == target
        p = output[-1:, :, :][mask]
        t = target[mask]
        loss = torch.sqrt(((p - t)**2).mean())
        return loss


class RmseMix(torch.nn.Module):
    def __init__(self):
        super(RmseMix, self).__init__()

    def forward(self, outTs, outC, tarTs, tarC):
        nts = outTs.shape[-1]
        nc = outC.shape[-1]
        # for time series
        lossTs = 0
        for k in range(nts):
            p0 = outTs[:, :, k]
            t0 = tarTs[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            if temp == temp:
                lossTs = lossTs + temp
        # for constant
        mask = tarC == tarC
        p = outC[mask]
        t = tarC[mask]
        lossC = torch.sqrt(((p - t)**2).nansum())
        return (lossTs+lossC)/(nts+nc)


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
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample + 1
        # minimize the opposite average NSE
        loss = -(losssum/nsample)
        return loss


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
