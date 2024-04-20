# from mpl_toolkits import basemap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from hydroDL import utils


def plotTS(ax, t, y, *, styLst=None, tBar=None, cLst='krbgcmy', legLst=None,
           sd=None, lineW=None, **kw):
    y = y if type(y) is list else [y]
    if sd is not None:
        ind = np.where(t >= sd)[0]
        t = t[ind]
        for k in range(len(y)):
            y[k] = y[k][ind]
    if lineW is None:
        lineW = [1.5 for k in y]
    for k, yy in enumerate(y):
        yy = y[k]
        # find out continuous / distinct
        if styLst is None:
            [_, _], ind = utils.rmNan([t, yy])
            r = len(ind)/(ind[-1]-ind[0]) if len(ind) > 0 else 0
            sty = '-' if r > 0.9 else '*'
        else:
            sty = styLst[k]
        legStr = None if legLst is None else legLst[k]
        ax.plot(t, yy, sty, color=cLst[k],
                label=legStr, linewidth=lineW[k], **kw)
    if tBar is not None:
        ylim = ax.get_ylim()
        tBar = [tBar] if type(tBar) is not list else tBar
        for tt in tBar:
            ax.plot([tt, tt], ylim, '-k')
    if legLst is not None:
        # ax.legend(loc='upper right', frameon=False)
        ax.legend(loc='upper right')
    ax.xaxis_date()
    return ax


def multiYrTS(axes, yrLst, t, dataPlot, cLst='rbk', styLst='--*', legLst=None):
    ty = t.astype('M8[Y]').astype(str).astype(int)
    indLst = list()
    ny = len(yrLst)
    for yr in yrLst:
        bp = np.in1d(ty, yr)
        ind = np.where(bp)[0]
        indLst.append(ind)
    for k in range(ny):
        # print(k)
        ind = indLst[k]
        for kk, x in enumerate(dataPlot):
            legStr = None if legLst is None else legLst[kk]
            axes[k].plot(t[ind], x[ind], styLst[kk],
                         color=cLst[kk], label=legStr)
        axes[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[k].xaxis.set_major_locator(mdates.MonthLocator([4, 8, 12]))
        axes[k].set_xlabel(yrLst[k])
        if k > 0:
            # axes[k].set_yticklabels([])
            axes[k].yaxis.set_visible(False)
    if legLst is not None:
        axes[-1].legend(loc='upper right')
    return axes


def multiTS(axes, t, dataPlot, labelLst=None, cLst='krbgcmy', styLst=None,
            tBar=None):
    # dataPlot - list [ndarray1[nt,ny], ndarray2[nt,ny], ...]
    # or just ndarray1[nt,ny]
    if type(dataPlot) is list:
        nd = dataPlot[0].shape[1]
    elif type(dataPlot) is np.ndarray:
        nd = dataPlot.shape[1]

    for k in range(nd):
        if type(dataPlot) is list:
            temp = [data[:, k] for data in dataPlot]
        elif type(dataPlot) is np.ndarray:
            temp = dataPlot[:, k]
        plotTS(axes[k], t, temp, cLst=cLst, styLst=styLst, tBar=tBar)
        axes[k].set_xlim(t[0], t[-1])
        if labelLst is not None:
            # titleInner(axes[k], labelLst[k])
            ax = axes[k].twinx()
            ax.set_ylabel(labelLst[k])
            ax.set_yticks([])
        if k != nd-1:
            axes[k].set_xticklabels([])
    return axes


def plotCDF(ax, x, cLst='rbkgcmy', legLst=None):
    x = x if type(x) is list else [x]
    for k in range(len(x)):
        xx = x[k]
        xS = sortData(xx)
        yS = np.arange(len(xS)) / float(len(xS) - 1)
        legStr = None if legLst is None else legLst[k]
        ax.plot(xS, yS, color=cLst[k], label=legStr)
    if legLst is not None:
        ax.legend(loc='bottom right', frameon=False)


def plotBox(ax, x, labLst=None, c='r'):
    temp = x
    if type(temp) is list:
        for kk in range(len(temp)):
            tt = temp[kk]
            if tt is not None and tt != []:
                tt = tt[~np.isnan(tt)]
                temp[kk] = tt
            else:
                temp[kk] = []
    else:
        temp = temp[~np.isnan(temp)]
    bp = ax.boxplot(temp, patch_artist=True, notch=True, showfliers=False)
    # ax.set_xticks(range(len(vLst)), vLst)
    if labLst is not None:
        _ = plt.setp(ax, xticks=range(1, len(labLst)+1), xticklabels=labLst)
    for kk in range(0, len(bp['boxes'])):
        _ = plt.setp(bp['boxes'][kk], facecolor=c)
    return ax


def plotHeatMap(ax, mat, labLst=None, fmt='{:.0f}', vRange=None):
    ny, nx = mat.shape
    ax.set_xticks(np.arange(nx))
    ax.set_yticks(np.arange(ny))
    if labLst is not None:
        if type(labLst[0]) is list:
            labX = labLst[1]
            labY = labLst[0]
        else:
            labX = labLst
            labY = labLst
        ax.set_xticklabels(labX)
        ax.set_yticklabels(labY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    if vRange is None:
        im = ax.imshow(mat)
    else:
        vmin, vmax = vRange
        im = ax.imshow(mat, vmin=vmin, vmax=vmax)
    for j in range(ny):
        for i in range(nx):
            text = ax.text(i, j, fmt.format(mat[j, i]),
                           ha="center", va="center", color="w")
    return ax


def plot121(ax, x, y, specP='b*', specL='k-', vR=None):
    _ = ax.plot(x, y, specP)
    if vR is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        vmin = np.min([xlim[0], ylim[0]])
        vmax = np.max([xlim[1], ylim[1]])
    else:
        [vmin, vmax] = vR
    _ = ax.plot([vmin, vmax], [vmin, vmax], specL)


def scatter121(ax, x, y, c, specL='k-', vR=None, size=None, cmap='viridis', **kw):
    if vR is None:
        vmax = None
        vmin = None
    else:
        vmin = vR[0]
        vmax = vR[1]
    out = ax.scatter(x, y, c=c, s=size, vmin=vmin, vmax=vmax, cmap=cmap,**kw)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = np.min([xlim[0], ylim[0]])
    vmax = np.max([xlim[1], ylim[1]])
    _ = ax.plot([vmin, vmax], [vmin, vmax], specL)
    return out


def sortData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return xSort


def titleInner(ax, titleStr, offset=0.1, top=True):
    if top:
        _ = ax.text(.5, 1-offset, titleStr, horizontalalignment='center',
                    transform=ax.transAxes)
    else:
        _ = ax.text(.5, offset, titleStr, horizontalalignment='center',
                    transform=ax.transAxes)


  
