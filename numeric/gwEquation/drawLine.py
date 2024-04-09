import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


class LineRegDrawer(object):
    def __init__(self, bbox=[0, 1, 0, 1], nPoint=100):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.axis(bbox)
        self.bbox = bbox
        self.nPoint = nPoint
        self.fig.show()

    def draw_line(self, order=2):
        xy = plt.ginput(-1)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        transformer = PolynomialFeatures(degree=order, include_bias=False)
        transformer.fit(np.array(x).reshape((-1, 1)))
        x_ = transformer.transform(np.array(x).reshape((-1, 1)))
        model = LinearRegression()
        model.fit(x_, y)
        xf = np.linspace(self.bbox[0], self.bbox[1], self.nPoint)
        xx = transformer.transform(xf.reshape(-1, 1))
        yy = model.predict(xx)        
        self.ax.plot(xf, yy)
        self.fig.canvas.draw()        
        return xf, yy


ld=LineRegDrawer()
x,y=ld.draw_line(order=3)