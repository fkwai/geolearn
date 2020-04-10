
import matplotlib.pyplot as plt
import numpy as np


def clickMap(funcMap, funcPoint):
    # funcMap - overall design and map plot
    # funcPoint - how to plot point given axes and index of point
    figM, axM, figP, axP, xLoc, yLoc = funcMap()
    if type(axM) is not list:
        axM = [axM]

    def onclick(event, figP, axP):
        xClick = event.xdata
        yClick = event.ydata
        if xClick is None or yClick is None:
            print('click on map plz')
            return
        iP = np.argmin(np.sqrt((xClick - xLoc)**2 + (yClick - yLoc)**2))
        for ax in axM:
            # for ax in temp:
            [p.remove() for p in reversed(ax.patches)]
            circle = plt.Circle([xLoc[iP], yLoc[iP]], 1,
                                color='black', fill=False)
            ax.add_patch(circle)
        for ax in axP.reshape(-1):
            ax.clear()
        funcPoint(iP, axP)
        figM.canvas.draw()
        figP.canvas.draw()
    figM.canvas.mpl_connect('button_press_event',
                            lambda event: onclick(event, figP, axP))
    figM.show()
    figP.show()


def boxPlot(data, label1=None, label2=None, cLst='rbkgcmy',
            figsize=(8, 6), sharey=True, widths=None, legOnly=False):
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
                if tt is not None and tt != []:
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
    if label2 is not None:
        ax.legend(bp['boxes'], label2, loc='best')
        if legOnly is True:
            ax.set_position([0, 0, 0.1, 1])
            ax.legend(bp['boxes'], label2, bbox_to_anchor=(1, 1))
    return fig
