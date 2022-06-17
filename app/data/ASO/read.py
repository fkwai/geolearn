from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

file = r'C:\Users\geofk\work\database\ASO\ASO_50M_SWE_USCOGE_20190407.tif'

im = Image.open(file)
imarray = np.array(im)
im.show()

fig, ax = plt.subplots(1, 1)
ax.imshow(imarray)
fig.show()



file = r'C:\Users\geofk\work\database\ASO\ASO_50M_SWE_USCOGE_20180331.tif'

im = Image.open(file)
imarray = np.array(im)
im.show()

fig, ax = plt.subplots(1, 1)
ax.imshow(imarray)
fig.show()