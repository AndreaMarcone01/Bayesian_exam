# Python script for point 1a: find parameters of the classification model

import numpy as np
from scipy.special import xlogy
from pdf_analysis import errors_around_peak

def gauss(x, mu, sigma):
    """Log normal model.
    
    Args:
        x (array): normal distributed variable, this is expected as log
        mu (float): mean of the distribution
        sigma (float): variance of the distribution
        
    Returns:
        pdf (array): probability density function, normalized to 1
    """

    pdf = 1/(np.sqrt(2 * np.pi)*sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    return pdf

def weighted_log_normal(x, theta):
    """Weighted sum of two log-normal.
    
    Args:
        x (array): independent variable
        theta (array): parameter array, in order [w, mu_1, sigma_1, mu_2, sigma_2]
        
    Returns:
        total model (array): probability density function, normalized to 1
    """

    w = theta[0]
    mu_1 = theta[1]
    sigma_1 = theta[2]
    mu_2 = theta[3]
    sigma_2 = theta[4]

    normal_1 = gauss(x, mu_1, sigma_1)
    normal_2 = gauss(x, mu_2, sigma_2)
    model = w * normal_1 + (1-w) * normal_2
    return model

def log_prior(theta, bounds):
    """Prior for the problem inside the bounds. Uniform for w, mu_1, mu_2, Jeffrey prior for sigma_1 and sigma_2
    
    Args:
        theta (array): parameters on which calculate the posterior [w, mu1, sigma1, mu2, sigma2]
        bounds (array): bound for each parameter, expected as [min, max]
        
    Returns:
        prior (float): prior for the sey theta 
    """
    
    prior = 0

    for i in range(theta.shape[0]):
        if theta[i] < bounds[i][0] or theta[i] > bounds[i][1]:
            return - np.inf
    
    # check if mu_1 < mu_2. System of penalty for mu_1 near mu_2?
    if theta[1] > theta[3]:
        return - np.inf
    
    # Jeffrey prior on the sigmas
    for i in [2,4]:
            if theta[i] <= 0:           # this should be removed by the bounds but better safe than sorry
                return -np.inf
            else:
                prior += - np.log(theta[i])
    
    return prior

def log_likelihood(theta, counts, center, model):
    """Poisson log likelihood for model.
    
    Args:
        theta (array): parameters of the model
        counts (array): counts of the histogram not normalized
        center (array): center of the bins of histogram
        model (function): model to use
        
    Returns:
        likelihood (float): log of the likelihood
    """

    N = np.sum(counts)
    dx = np.diff(center)[0]
    expected_count = N * model(center, theta) * dx

    if np.any((expected_count == 0) & (counts > 0)):       # if expected == 0 we have a nan problem, but if also counts == 0 it's right
            return -np.inf                                          # in the case that the model sees 0 counts but in realty there are return -inf
        
    log_like = xlogy(counts, expected_count) - expected_count
    return np.sum(log_like)

def log_posterior(theta, counts, center, model, bounds):
    """Log posterior for model.
    
    Args:
        theta (array): parameters of the model
        counts (array): counts of the histogram not normalized
        center (array): center of the bins of histogram
        model (function): model to use
        bounds (array): bound for each parameter, expected as [min, max]
        
    Returns:
        posterior (array): log of the posterior
    """
    
    posterior = log_prior(theta, bounds) + log_likelihood(theta, counts, center, model)
    return posterior

def proposed_distribution(x, bounds, rng, blind = True):
    """Proposed distribution for the MCMC algorithm.
    
    Args:
        x (array): input of the distribution
        bound (array): bound for each parameter, expected as [min, max]. Used to define covariance
        rng (np.random.default_rng): default rng for reproduce results
        blind (boolean): decide if run with a blind covariance or use a already note covariance
        
    Returns:
        pdf (array): proposed distribution values
    """

    d = x.shape[0]
    if blind == False:
        fname = main_dir+"\\samples_covariance.txt"
        if os.path.isfile(fname) == True: 
            covariance = np.loadtxt(fname)
        else:
            print("Covariance file not found!")
            scales = 0.05 * np.diff(bounds)[:,0]
            covariance = np.diag(scales**2) # diagonal matrix of dimension d*d
    
    else:
        scales = 0.05 * np.diff(bounds)[:,0]
        covariance = np.diag(scales**2) # diagonal matrix of dimension d*d
    
    pdf = rng.multivariate_normal(np.zeros(d), covariance)
    return pdf

def metropolis_hastings(theta0, postpdf, counts, center, model, bounds, rng, blind, n = 10000):
    """Metropolis hastings algorithm to sample the posterior.
    
    Args:
        theta0 (array): initial point for the chain
        postpdf (function): posterior to use for the chain
        counts (array): counts of the histogram not normalized
        center (array): center of the bins of histogram
        model (function): model to use for posterior
        bounds (array): bound for each parameter, expected as [min, max]
        rng (np.random.default_rng): default rng for reproduce results
        blind (boolean): decide if run with a blind covariance or use a already note covariance
        n (float): length of the chain, default to 10000
    
    Returns:
        samples (array): samples of the parameters
    """

    d = theta0.shape[0]
    logP0 = postpdf(theta0, counts, center, model, bounds)
    samples = np.zeros((n,d), dtype=np.float64)
    logP = np.zeros(n, dtype=np.float64)
    accepted = 0
    rejected = 0

    for i in range(n):
        theta_try = theta0 + proposed_distribution(theta0, bounds, rng, blind)
        logP_try = postpdf(theta_try, counts, center, model, bounds)
        
        if logP_try - logP0 > np.log(np.random.uniform(0,1)):
            samples[i,:] = theta_try
            logP[i] = logP_try
            theta0 = theta_try
            logP0 = logP_try
            accepted += 1
        else:
            samples[i,:] = theta0
            logP[i] = logP0
            rejected += 1
        
        print(f"Iteration {i}, acceptance = {accepted/(accepted+rejected)}")

    return samples, logP

def autocorrelation(x, norm = True):
    """Find the autocorrelation of x
    
    Args:
        x (array): array on which compute the autocorrelation
        norm (boolean): decide if normalize or not, default to True
        
    Returns:
        autocorr (array): array of autocorrelation
    """

    f = np.fft.fft(x - np.mean(x), n = 2*len(x))
    f_con = np.conjugate(f)
    corr = np.real(np.fft.ifft(f * f_con)[:len(x)])

    if norm == True:
        corr /= corr[0]
        
    return corr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(1313) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T90, d_log_T90, log_HR = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)

    fig1 = plt.figure("logT-logHR plane")
    plt.scatter(log_T90, log_HR, marker = '.', label = 'Data')
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("$\\log(HR)$")
    plt.tight_layout()
    

    fig1 = plt.figure("logT-logHR plane and marginals")
    ax = fig1.add_subplot(2,2,3)
    axT = fig1.add_subplot(2,2,1, sharex = ax)
    axH = fig1.add_subplot(2,2,4, sharey = ax)
    axT.hist(log_T90, bins=50)
    axH.hist(log_HR, bins=50, orientation="horizontal")
    ax.scatter(log_T90, log_HR, marker = '.', label = 'Data')
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("$\\log(HR)$")
    plt.tight_layout()
    plt.show()