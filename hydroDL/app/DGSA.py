# Author: David Zhen Yin, Yizheng Wang
# Contact: yinzhen@stanford.edu, yizhengw@stanford.edu
# Date: August 22, 2019
import numpy as np
from tqdm import tqdm
# from KMedoids import KMedoids
import pandas as pd
import warnings
from sklearn.metrics.pairwise import euclidean_distances

# warnings.filterwarnings("ignore")


def DGSA_light(parameters, responses, ParametersNames=None, n_clsters=3, n_boots=3000):
    '''
    Main function of DGSA light version
    Parameters
    ------------
    parameters: input model parameters, 2D array (#samples, #parameters)
    responses: model responses to the input parameters, 2D array (#samples, #responses)
    ParametersNames (optional): name of the input model parameters, 1D list, default = ['p1', 'p2', ....]
    n_clsters (optional): number of KMedoids clusters to classify the model responses, default = 3
    n_boots (optional): number of boostrap resamplings, default = 3000

    Output
    ------------
    dgsa_measures_main:  main sensitivity of parameters measured by DGSA, (pd.DataFrame)data frame. 
    '''
    n_samples, n_parameters = parameters.shape[0], parameters.shape[1]
    '''STEP 1. K-Medoids clustering'''
    OK = False
    while not OK:
        try:
            model = KMedoids(n_clusters=n_clsters)
            Medoids, clsters = model.fit(responses, plotit=False)
            OK = True
        except:
            OK = False
    '''STEP 2. Calculate L1-Norm distance between sample distribution and cluster distributions'''
    '''STEP 2.1 Calucate the CDF of the original parameters'''
    percentiles = np.arange(100)
    cdf_parameters = np.percentile(parameters, percentiles, axis=0)

    '''STEP 2.2 Calculate the L1 norm for the clusters & Run bootstrap sampling'''
    def L1norm_cls(k):
        '''Define function to calculate L1-norm for clustered parameters'''
        parameters_cls = parameters[clsters[k]]
        L1norm_clster[k, :] = np.sum(abs(np.percentile(
            parameters_cls, percentiles, axis=0) - cdf_parameters), axis=0)
        return L1norm_clster[k, :]
    L1norm_clster = np.zeros((n_clsters, n_parameters))
    [L1norm_cls(n_c) for n_c in range(n_clsters)]

    '''STEP 2.3 Calculate the L1 norm for the n bootstraps'''
    def L1norm_Nboots(k, p):
        '''Define function to calculate L1-norm distances for N boostrap sampling'''
        parameters_Nb = parameters[np.random.choice(
            len(parameters), len(clsters[k]), replace=False)]
        L1norm_Nb[p, k, :] = np.sum(
            abs(np.percentile(parameters_Nb, percentiles, axis=0) - cdf_parameters), axis=0)
        return L1norm_Nb[p, k, :]
    L1norm_Nb = np.zeros((n_boots, n_clsters, n_parameters))
    [[L1norm_Nboots(n_c, p) for n_c in range(n_clsters)]
     for p in tqdm(range(n_boots))]

    '''STEP 3. Calculate main DGSA measurements'''
    dgsa_measures_cls = L1norm_clster/(np.percentile(L1norm_Nb, 95, axis=0))
    dgsa_measures_main = np.max(dgsa_measures_cls, axis=0)

    if ParametersNames == None:
        dgsa_measures_main = pd.DataFrame(
            dgsa_measures_main, ['p{}'.format(i) for i in range(1, n_parameters+1)])
    else:
        dgsa_measures_main = pd.DataFrame(dgsa_measures_main, ParametersNames)

    return dgsa_measures_main


def _get_init_centers(n_clusters, n_samples):
    '''return random points as initial centers'''
    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0, n_samples)
        if not _ in init_ids:
            init_ids.append(_)
    return init_ids


def _get_cost(dist_meds, currentMedoids):
    '''return total cost and cost of each cluster
    -----
    currentMedoids: the current Medoids
    dist_meds: paird distances between all data points and each Medoid
    '''
    costs = np.zeros(len(currentMedoids))
    dis_min = np.min(dist_meds, axis=0)
    for i in range(len(currentMedoids)):
        clst_mem_ids = np.where(dist_meds[i] == dis_min)[0]
        costs[i] = np.sum(dist_meds[i][clst_mem_ids])
    return np.sum(costs)


def _kmedoids_run(X, n_clusters, max_iter, tolerance):
    '''
    Main function for runing the k-medoids clustering
    -------------
    X: the input data ndarray for k-medoids clustering, (#samples, #features)
    n_cluster: number of clusters
    max_iter: maximum number of clusters
    torlerance: the tolerance to stop the iterations, in percentage; 
                i.e.: if tolerance=0.01, it means if the cost function decrease is less than 1%, the iteraction will stop.
    '''

    n_samples = len(X)
    '''Calcuate the paired eucledian distance '''
    dist_mat = euclidean_distances(X)
    ''' Initialize the medoids'''
    currentMedoids = np.asarray(_get_init_centers(n_clusters, n_samples))

    '''Calcualte the total cost of the initial medoids'''
    costs_iters = []
    dist_meds = dist_mat[currentMedoids]
    tot_cos = _get_cost(dist_meds, currentMedoids)
    costs_iters.append(tot_cos)
    cc = 0

    for i in range(max_iter):
        dist_meds = dist_mat[currentMedoids]
        '''Associate  each data point to the closest medoid
            And calcualte the total cost'''
        tot_cos = _get_cost(dist_meds, currentMedoids)
        '''Get new mediods o'''
        newMedoids = []
        for j in range(n_clusters):
            o = np.random.choice(n_samples)
            if (not o in currentMedoids and not o in newMedoids):
                newMedoids.append(o)
        newMedoids = np.asarray(newMedoids).astype(int)
        dist_meds_ = dist_mat[newMedoids]
        tot_cos_ = _get_cost(dist_meds_, newMedoids)
        '''Swap newmediods with the current mediod if cost decreases'''
        if (tot_cos_ - tot_cos) < 0:
            currentMedoids = newMedoids
            costs_iters.append(tot_cos_)
            cc = +1
            if abs(costs_iters[cc]/costs_iters[cc-1]-1) < tolerance:
                '''Associated  data points to the final calucated medoids (reached by tolerance)'''
                clsts_membr_ids = []
                dis_min = np.min(dist_meds, axis=0)
                for k in range(n_clusters):
                    clst_mem_ids = np.where(dist_meds[k] == dis_min)[0]
                    clsts_membr_ids.append(clst_mem_ids)

                return currentMedoids, clsts_membr_ids, costs_iters
                break

    costs_iters = np.asarray(costs_iters)
    '''Associated  data points to the final calucated medoids (reached by maximum iters)'''
    clsts_membr_ids = []
    dist_meds = dist_mat[currentMedoids]
    dis_min = np.min(dist_meds, axis=0)
    for k in range(n_clusters):
        clst_mem_ids = np.where(dist_meds[k] == dis_min)[0]
        clsts_membr_ids.append(clst_mem_ids)

    return currentMedoids, clsts_membr_ids, costs_iters


class KMedoids(object):
    '''
    Main API of KMedoids Clustering
    Parameters
    --------
        X: the input ndarray data for k-medoids clustering, (#samples, #features)
        n_clusters: number of clusters
        max_iter: maximum number of iterations
        tolerance:  the tolerance to stop the iterations, in percentage; 
                    i.e.: if tolerance=0.01, it means if the cost function decrease is less than 1%, the iteraction will stop.
    Attributes
    --------

        Medoids   :  cluster Medoids id
        costs_itr   :  array of costs for each effective iterations
        clst_membr_ids   :  each cluster members' sample-ids in the input ndarray X 

    Methods
    -------
        model = KMedoids(n_cluster = #, max_iter=#<opertional>, tolerance==#<opertional>)
        Medoids, cluster_id = model.fit(X, plotit=True/False): fit the model, 
                                                    it returns the center medoids id in the ndarray X, 
                                                    and  each cluster members' sample-ids in X.
    '''

    def __init__(self, n_clusters, max_iter=10000, tolerance=0.005):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, plotit=True):
        '''Run the main k-medoids function'''
        Medoids, clst_membr_ids, costs_itr = _ = _kmedoids_run(
            X, self.n_clusters, self.max_iter, self.tolerance)
        ''' Plot or not'''
        if plotit:
            fig, ax = plt.subplots(1, 1)
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            if self.n_clusters > len(colors):
                raise ValueError('we need more colors')
            for i in range(self.n_clusters):
                X_c = X[clst_membr_ids[i]]
                ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.7, s=30)
                ax.scatter(X[Medoids[i], 0], X[Medoids[i], 1], c=colors[i],
                           alpha=1., linewidth=1.5, edgecolor='k', s=200, marker='*')

        return Medoids, clst_membr_ids
