
from scipy import sparse
import os
import numpy as np
dirData = r'C:\Users\geofk\work\waterQuality\trainDataFull\Q90ref'

cFile = os.path.join(dirData, 'c.npy')
qFile = os.path.join(dirData, 'q.npy')
cFile2 = os.path.join(dirData, 'c.npz')
qFile2 = os.path.join(dirData, 'q.npz')

c = np.load(cFile)
q = np.load(qFile)

len(np.where(~np.isnan(c))[0])
len(np.where(~np.isnan(q))[0])

sc = sparse.csc_matrix(c[:, :, 0])
sparse.save_npz(cFile2, sc)

np.savez_compressed(qFile2, q=q)
np.save(qFile, q)

q.save(qFile)

np.savez_compressed(cFile2, c=c)

cc=np.load(cFile2)

# savez_compressed is the one I need!!!