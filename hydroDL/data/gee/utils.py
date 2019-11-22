
import ee

def bb2ee(bb):
    [y1, x1, y2, x2]=bb
    rect = ee.Geometry.Rectangle([x1, y1, x2, y2])
    return rect

def t2ee(t):
    tt = ee.Date.fromYMD(t.year, t.month, t.day)
    return tt

