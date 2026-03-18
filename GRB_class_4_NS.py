# Python script to run the NS for point 4. Raynest doesn't work on windows, run this in Ubuntu terminal

import numpy as np
from scipy.special import xlogy
import raynest.model

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

def three_weighted_log_normal(x, theta):
    """Weighted sum of 3 log-normal.
    
    Args:
        x (array): independent variable
        theta (array): parameter array, in order [w, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, w_2]
        
    Returns:
        total model (array): probability density function, normalized to 1
    """

    w_1 = theta[0]
    mu_1 = theta[1]
    sigma_1 = theta[2]
    mu_2 = theta[3]
    sigma_2 = theta[4]
    mu_3 = theta[5]
    sigma_3 = theta[6]
    w_2 = theta[7]

    normal_1 = gauss(x, mu_1, sigma_1)
    normal_2 = gauss(x, mu_2, sigma_2)
    normal_3 = gauss(x, mu_3, sigma_3)
    model = w_1 * normal_1 + w_2 * normal_2 + (1-w_1-w_2) * normal_3
    return model

class FunctionalModel(raynest.model.Model):
    def __init__(self, counts, center, model, bounds, names):
        self.counts = counts        # counts in a bin
        self.center = center        # centers of the bins
        self.model  = model         # model to evaluate
        self.bounds = bounds        # bounds on the parameters
        self.names  = names
        self.N      = np.sum(counts)
        self.dx     = np.diff(center)[0]

    def log_prior(self, p):
        # p stands for live point so it's the array of parameters
        
        # check the bounds
        for i in range(p.shape[0]):
            if p[i] < self.bounds[i][0] or p[i] > self.bounds[i][1]:
                return -np.inf
            
        # in the three normal case constrain w_1+w_2<1
        if p.shape[0]>5:
            if p[0] + p[7] > 1:
                return -np.inf
            
        # penalty for having mu_1 > mu_2 or mu_2 > mu_3
        for i in [1,3]:
            if i + 3 < p.shape[0]:
                if p[i] > p[i+2]:
                    return -np.inf
        
        # Jeffrey prior on the sigmas
        prior = 0.0
        for i in [2,4,6]:
            if i < p.shape[0]:
                if p[i] <= 0:           # this should be removed by the bounds but better safe than sorry
                    return -np.inf
                else:
                    prior += -np.log(p[i])

        return 0.0
    
    def log_likelihood(self, p):
        expected_count = self.N * self.model(self.center, p) * self.dx
        if np.any((expected_count == 0) & (self.counts > 0)):       # if expected == 0 we have a nan problem, but if also counts == 0 it's right
            return -np.inf                                          # in the case that the model sees 0 counts but in realty there are return -inf
        
        log_like = xlogy(self.counts, expected_count) - expected_count
        log_like = np.sum(log_like)
        return log_like


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(111) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T90, d_log_T90, Hardness_R = np.loadtxt(main_dir+"/Data/GRB_data.txt", unpack=True)
    
    nbins = 50
    bins = np.linspace(-4, 7, nbins)
    hist, edges = np.histogram(log_T90, bins = bins, density = True)    # used in plot
    counts, _ = np.histogram(log_T90, bins = bins)                      # used for the real analysis
    center = (edges[1:] + edges[:-1])*0.5

    name2 = ["w", "mu_1", "sigma_1", "mu_2", "sigma_2"]
    bounds2 = np.array([[0.0,1.0], [-4.0,7.0], [0.01,3.0] , [-4.0, 7.0], [0.01,3.0]])

    name3 = ["w_1", "mu_1", "sigma_1", "mu_2", "sigma_2", "mu_3", "sigma_3", "w_2"]
    # updated bounds given the precedent results
    bounds3 = np.array([[0.0,1.0], 
                        [-4.0, 2.0], [0.01,3.0],     # first gaussian on the short GRB
                        [-4.0, 7.0], [0.01,3.0],     # intermediate? Less known
                        [0.0, 7.0], [0.01,3.0],      # third on long GRB
                        [0.0,1.0]])
        
    # try to initialise things
    
    two_class_model = FunctionalModel(counts, center, weighted_log_normal, bounds2, name2)
    three_class_model = FunctionalModel(counts, center, three_weighted_log_normal, bounds3, name3)

    run2  = True
    run3  = True
    nlive = 2000

    if run2 == True:
        work = raynest.raynest(two_class_model, verbose=2,                  # model on which infer, output on screen and memory (how much the function speaks)
                               nnest=1,                                     # parallelize: number of parallel algorithm
                               nensemble=6, nslice=0, nhamiltonian=0,       # method of replacing the live points
                               nlive=nlive, maxmcmc=5000,                   # number of live points (> 2*ndim + 1), max number of MCMC willing to take to have the new live point
                               resume=0, periodic_checkpoint_interval=100,  # resume True and save the situation of algorithm every periodic_time 
                               output='J_two_class_output')                   # where to save the results 
        work.run()

    if run3 == True:
        work = raynest.raynest(three_class_model, verbose=2,
                               nnest=1,
                               nensemble=6, nslice=0, nhamiltonian=0,
                               nlive=nlive, maxmcmc=5000,
                               resume=0, periodic_checkpoint_interval=100,
                               output='J_three_class_output')
        work.run()