import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
