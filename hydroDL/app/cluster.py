import numpy as np


def fullPam(matD, medoid):
    distM = np.nanmean(np.min(matD[medoid, :], axis=0))
    nk = medoid.shape[0]
    n = matD.shape[0]
    for i in range(nk):
        for k in range(n):
            mNew = medoid.copy()
            mNew[i] = k
            distT = np.nanmean(np.min(matD[mNew, :], axis=0))
            if distT < distM:
                return mNew, distT
    return medoid, distM


def kmedoid(matD, nk):
    n = matD.shape[0]
    medoid = np.random.randint(0, n, nk)
    mNew, distM = fullPam(matD, medoid)
    iter = 0
    while not (mNew == medoid).all():
        iter = iter + 1
        # print(iter)
        medoid = mNew
        mNew, distM = fullPam(matD, medoid)
    outMedoid = np.sort(medoid)
    outCluster = np.argmin(matD[outMedoid, :], axis=0)
    return outMedoid, outCluster


def silhouette(matD, medoid, cluster):
    nk = len(medoid)
    distS = np.zeros([len(cluster), nk])
    matSil = np.zeros([len(cluster), 2])
    for i in range(len(cluster)):
        for k in range(nk):
            distS[i, k] = np.mean(matD[i, cluster == k])
        mask = np.arange(nk) == cluster[i]
        a = np.mean(distS[i, mask])
        b = np.min(distS[i, ~mask])
        matSil[i, :] = [a, b]
    scoreSil = (matSil[:, 1] - matSil[:, 0]) / matSil[:, 1]
    return np.mean(scoreSil)
