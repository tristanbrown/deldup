import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

def gauss(x,mu,sigma,A):
    """A Gaussian distribution model."""
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def trimodal(x,mu,sigma1,A1,sigma2,A2,sigma3,A3):
    """A trimodal Gaussian distribution model, centered at mu/2, mu, 3mu/2"""
    return gauss(x,mu/2,sigma1,A1)+gauss(x,mu,sigma2,A2)+gauss(x,mu/2*3,sigma3,A3)

def fit_deldup(data, model=trimodal, visual=False):
    """Try fitting a trimodal, or if it fails, unimodal Gaussian model."""
    y,x,_=plt.hist(data,100,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

    mu,std,amp = data.mean(),data.std(),data.max()
    if model == trimodal:
        expected=(mu,std,amp/10,std,amp,std,amp/10)
    elif model == gauss:
        expected=(mu,std,amp)
    try:
        params,cov=curve_fit(model,x,y,expected)
    except RuntimeError as ex:
        if 'Optimal parameters not found' in str(ex) and model == trimodal:
            cov = [np.inf]
        else:
            return None
    if np.inf in cov:
        return fit_deldup(data, model=gauss, visual=visual)
    sigma=np.sqrt(np.diag(cov))
    if visual:
        plt.plot(x,model(x,*params),color='red',lw=3,label='model')
        plt.legend()
        plt.show()
    return pd.DataFrame(data={'params':params,'sigma':sigma},index=model.__code__.co_varnames[1:])

def fit_gaussian(data, visual=False):
    """Try fitting a trimodal, or if it fails, unimodal Gaussian model."""
    if data is None:
        return None
    elif isinstance(data, list):
        return [fit_gaussian(clust, visual) for clust in data]
    plt.figure()
    y,x,_=plt.hist(data,100,alpha=.3,label='data')
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

    mu,std,amp = data.mean(),data.std(),data.max()
    expected=(mu,std,amp)
    try:
        params,cov=curve_fit(gauss,x,y,expected)
    except RuntimeError as ex:
        if 'Optimal parameters not found' in str(ex):
            return None
        else:
            raise
    sigma=np.sqrt(np.diag(cov))
    result = pd.DataFrame(data={'params':params,'sigma':sigma},index=gauss.__code__.co_varnames[1:])
    if visual:
        plt.plot(x,gauss(x,*params),color='red',lw=3,label='model')
        plt.legend()
        plt.show()
        print(result)
    return result

def model_cn(data, visual=False):
    if not resolvable(data):
        return None
    if empty_tails(data):
        return data
    clusters = cluster(data, k=3)
    if not valid_clusters(clusters):
        clusters = cluster_bisect(data)
        if not valid_clusters(clusters):
            return data
    return clusters
    
def resolvable(data):
    """Check if CN=1,2,3 can be resolved, based on 2x stdev."""
    return (data.std() / data.mean() < 0.25)

def empty_tails(data):
    """Check if the regions containing CN=1 and CN=3 are completely empty."""
    mu = data.mean() / 2
    left_tail = data[data < 1.5 * mu]
    right_tail = data[data > 2.5 * mu]
    return left_tail.empty and right_tail.empty

def cluster(data, k, start=None):
    mu = data.mean()
    if k == 3 and start is None:
        start = [mu/2, mu, mu/2*3]
    elif k == 2 and start is None:
        start = [data.min(), data.max()]
    init = np.array(start).reshape(-1, 1)
    arr = np.array(data).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, init=init).fit(arr)
    labels = kmeans.labels_
    clusters = [data[labels == label] for label in range(k)]
    return clusters

def cluster_bisect(data):
    """Find 3 clusters via a bisecting method."""
    mu = data.mean()
    std = data.std()
    left_data = data[data < mu - 1.5*std]
    right_data = data[data > mu + 1.5*std]
    left_clust = cluster(left_data, 2, [mu/2, mu])[0]
    right_clust = cluster(right_data, 2, [mu, mu*3/2])[1]
    mid_clust = data[(data > left_clust.max())&(data < right_clust.min())]
    return [left_clust, mid_clust, right_clust]

def valid_clusters(clusters):
    """Validate the clusters based on CN and skew."""
    if clusters[1].skew() > 1:
        return False
    mu1,mu2,mu3 = (cluster.mean() for cluster in clusters)
    std = clusters[1].std()
    for est_mu in (2*mu1, mu3/3*2):
        if est_mu < mu2 - std or est_mu > mu2 + std:
            return False
    return True
