
import matplotlib.pyplot as plt
import numpy as np


def clickMap(funcMap, funcPoint):
    # funcMap - overall design and map plot
    # funcPoint - how to plot point given axes and index of point
    figM, axM, figP, axP, xLoc, yLoc = funcMap()

    def onclick(event, figP, axP):
        xClick = event.xdata
        yClick = event.ydata
        if xClick is None or yClick is None:
            print('click on map plz')
            return
        iP = np.argmin(np.sqrt((xClick - xLoc)**2 + (yClick - yLoc)**2))
        for temp in axM:
            for ax in temp:
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
