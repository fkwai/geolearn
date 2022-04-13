
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import axplot
from hydroDL import utils
import matplotlib.gridspec as gridspec


def clickMap(funcMap, funcPoint):
    # funcMap - overall design and map plot
    # funcPoint - how to plot point given axes and index of point
    figM, axM, figP, axP, xLoc, yLoc = funcMap()
    if type(axM) is not np.ndarray:
        axM = np.array([axM])

    def onclick(event, figP, axP):
        xClick = event.xdata
        yClick = event.ydata
        if xClick is None or yClick is None:
            print('click on map plz')
            return
        iP = np.nanargmin(np.sqrt((xClick - xLoc)**2 + (yClick - yLoc)**2))
        for ax in axM.flatten():
            # for ax in temp:
            [p.remove() for p in reversed(ax.patches)]
            circle = plt.Circle([xLoc[iP], yLoc[iP]], 1,
                                color='black', fill=False)
            ax.add_patch(circle)
        if type(axP) is not np.ndarray:
            axP.clear()
        else:
            for ax in axP.reshape(-1):
                ax.clear()
        funcPoint(iP, axP)
        figM.canvas.draw()
        figP.canvas.draw()
    figM.canvas.mpl_connect('button_press_event',
                            lambda event: onclick(event, figP, axP))
    figM.show()
    figP.show()
    return figM, figP


def clickMulti(funcM, funcP, funcT=None, circleSize=None):
    figM, axM, figP, axP, xMat, yMat, labelLst = funcM()
    if type(axM) is not np.ndarray:
        axM = np.array([axM])
    if circleSize is None:
        xR = np.nanmax(xMat, axis=0)-np.nanmin(xMat, axis=0)
        yR = np.nanmax(yMat, axis=0)-np.nanmin(yMat, axis=0)
        circleSize = np.min([xR, yR], axis=0)/10

    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        label = event.inaxes.get_label()
        iM = labelLst.index(label)
        xx = xMat[:, iM]
        yy = yMat[:, iM]
        iP = np.nanargmin(np.sqrt((xClick - xx)**2 + (yClick - yy)**2))
        for k, (ax, labelT) in enumerate(zip(axM.flatten(), labelLst)):
            [p.remove() for p in reversed(ax.patches)]
            xc = xMat[iP, k]
            yc = yMat[iP, k]
            color = 'red' if labelT == label else 'black'
            circle = plt.Circle([xc, yc], circleSize[k],
                                color=color, fill=False)
            ax.add_patch(circle)
        for ax in axP:
            ax.clear()
        funcP(axP, iP, iM)
        if funcT is not None:
            title = funcT(iP, iM)
            figP.suptitle(title)
        figM.canvas.draw()
        figP.canvas.draw()
    figM.canvas.mpl_connect('button_press_event',
                            lambda event: onclick(event))
    figM.show()
    figP.show()
    return figM, figP


def boxPlot(data, label1=None, label2=None, cLst='rbkgcmy',
            figsize=(8, 6), sharey=True, widths=None, legOnly=False,
            yRange=None):
    nc = len(data)
    if legOnly is True:
        nc = 1
    fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize)

    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        temp = data[k]
        if type(temp) is list:
            for kk in range(len(temp)):
                tt = temp[kk]
                if tt is not None and len(tt) != 0:
                    tt = tt[~np.isnan(tt)]
                    temp[kk] = tt
                else:
                    temp[kk] = []
        else:
            temp = temp[~np.isnan(temp)]
        bp = ax.boxplot(temp, patch_artist=True, notch=True,
                        showfliers=False, widths=widths)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=cLst[kk])
        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        ax.set_xticks([])
        # ax.ticklabel_format(axis='y', style='sci')
        if yRange is not None:
            ax.set_ylim(yRange)
    if label2 is not None:
        ax.legend(bp['boxes'], label2, loc='best')
        if legOnly is True:
            ax.set_position([0, 0, 0.1, 1])
            ax.legend(bp['boxes'], label2, bbox_to_anchor=(1, 1))
    return fig, axes


def tsYr(t, y, cLst='rbkgcmy', figsize=(12, 4), showCorr=False):
    y = y if type(y) is list else [y]
    yrAll = pd.to_datetime(t).year
    yrLst = yrAll.unique().tolist()
    ny = len(yrLst)
    fig, axes = plt.subplots(ncols=ny, sharey=True, figsize=figsize)
    fig.subplots_adjust(wspace=0)
    for iYr, yr in enumerate(yrLst):
        ind = np.where(yrAll == yr)[0]
        _ = axplot.plotTS(axes[iYr], t[ind], [v[ind] for v in y], cLst=cLst)
        _ = axes[iYr].set_xlim(np.datetime64(
            str(yr)), np.datetime64(str(yr+1)))
        _ = axes[iYr].set_xticks([])
        corr = np.corrcoef(utils.rmNan([v[ind]
                                        for v in y], returnInd=False))[0, 1]
        if showCorr is True:
            _ = axes[iYr].set_xlabel('{}\n{:.2f}'.format(yr, corr))
        else:
            _ = axes[iYr].set_xlabel('{}'.format(yr))
    return fig


def scatter121Batch(xMat, yMat, cMat, labelLst, nXY,
                    optCb=1, cR=None, ticks=None, s=15,
                    figsize=None, titleLst=None, cmap='viridis'):
    # plot nx*nx 121 maps
    # xMat, yMat - [#data, #features]
    # optCb - 0 no colorbar; 1 shared color bar; 2 - individual colorbar
    figM = plt.figure(figsize=figsize)
    [nfx, nfy] = nXY
    rCb = 10  # colorbar will be of width 1/rCb of subplot
    if optCb == 0:
        gsM = gridspec.GridSpec(nfy, nfx)
    elif optCb == 1:
        gsM = gridspec.GridSpec(nfy*rCb, nfx*rCb+1)
        if cR is None:
            cR = [np.nanmin(cMat), np.nanmax(cMat)]
    elif optCb == 2:
        gsM = gridspec.GridSpec(nfy*rCb, nfx*(rCb+1))
    # plot scatter
    axM = list()
    for k, label in enumerate(labelLst):
        j, i = utils.index2d(k, nfy, nfx)
        if optCb == 0:
            ax = figM.add_subplot(gsM[j:j+1, i:i+1])
        elif optCb == 1:
            ax = figM.add_subplot(gsM[(j)*rCb:(j+1)*rCb, i*rCb:(i+1)*rCb])
        elif optCb == 2:
            ax = figM.add_subplot(
                gsM[(j)*rCb:(j+1)*rCb, i*(rCb+1):i*(rCb+1)+rCb])
        ax.set_label(label)
        axM.append(ax)
        c = cMat[:, k] if cMat.ndim == 2 else cMat
        sc = axplot.scatter121(
            ax, xMat[:, k], yMat[:, k], c, vR=cR, size=s, cmap=cmap)
        corr = utils.stat.calCorr(xMat[:, k], yMat[:, k])
        titleStr = '{} {:.2f}'.format(label, corr)
        axplot.titleInner(ax, titleStr)
        if ticks is not None:
            _ = ax.set_xlim([ticks[0], ticks[-1]])
            _ = ax.set_ylim([ticks[0], ticks[-1]])
            _ = ax.set_yticks(ticks)
            _ = ax.set_xticks(ticks)
            if i != 0:
                _ = ax.set_yticklabels([])
            if j != nfy:
                _ = ax.set_xticklabels([])
            figM.subplots_adjust(wspace=0, hspace=0)
        if optCb == 2:
            cax = figM.add_subplot(
                gsM[(j)*rCb:(j+1)*rCb, (i+1)*(rCb+1)-1])
            figM.colorbar(sc, cax=cax)
    if optCb == 1:
        cax = figM.add_subplot(gsM[:, -1])
        figM.colorbar(sc, cax=cax)
    return figM, axM


def multiTS(t, dataPlot, labelLst=None, styLst=None, cLst='krbgcmy',
            tBar=None, lineW=None, figsize=None):
    # dataPlot - list [ndarray1[nt,ny], ndarray2[nt,ny], ...]
    # or just ndarray1[nt,ny]
    if type(dataPlot) is list:
        nd = dataPlot[0].shape[1]
    elif type(dataPlot) is np.ndarray:
        nd = dataPlot.shape[1]
    fig, axes = plt.subplots(nd, 1, sharex=True, figsize=figsize)
    if nd > 1:
        axplot.multiTS(axes, t, dataPlot, labelLst=labelLst,
                       cLst=cLst, tBar=tBar, styLst=styLst)
    else:
        axplot.plotTS(axes, t, dataPlot, cLst=cLst, tBar=tBar, lineW=lineW)
    fig.subplots_adjust(hspace=0)
    return fig, axes
