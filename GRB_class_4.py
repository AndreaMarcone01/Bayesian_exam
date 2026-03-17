import numpy as np
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

    def log_prior(self, p):
        # p stands for live point so it's the array of parameters
        prior = np.zeros(p.shape[0])

        for i in range(p.shape[0]):
            if p[i] < self.bounds[i][0] or p[i] > self.bounds[i][1]:
                prior[i] = - np.inf

        # penalty for having mu_1 > mu_2 or mu_2 > mu_3
        for i in [1,3]:
            if i + 3 < p.shape[0]:
                if p[i] > p[i+2]:
                    prior[i] = -np.inf

        prior = np.sum(prior)
        return prior
    
    def log_likelihood(self, p):
        N = np.sum(self.counts)
        dx = np.diff(self.center)[0]
        expected_count = N * self.model(center, p) * dx
        log_like = counts * np.log(expected_count) - expected_count
        return np.sum(log_like)


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
    bounds2 = np.array([[0.0,1.0], [-4.0,7.0], [0.0,3.0] , [-4.0, 7.0], [0.0,3.0]])

    name3 = ["w_1", "mu_1", "sigma_1", "mu_2", "sigma_2", "mu_3", "sigma_3", "w_2"]
    # updated bounds given the precedent results
    bounds3 = np.array([[0.0,1.0], 
                        [-4.0, 0.0], [0.0,3.0],
                        [-4.0, 7.0], [0.0,3.0],
                        [1.0, 7.0], [0.0,3.0], 
                        [0.0,1.0]])
    
    xx = np.linspace(-4,7,100)
    par_val = np.array([0.1, -1, 0.5, 1, 1, 3.5, 1, 0.2])
    pdf_f = three_weighted_log_normal(xx, par_val)
    w_normal_1 = par_val[0] * gauss(xx, par_val[1], par_val[2])
    w_normal_2 = par_val[7] * gauss(xx, par_val[3], par_val[4])
    w_normal_3 = (1-par_val[0]-par_val[7]) * gauss(xx, par_val[5], par_val[6])

    """# plot the data
    plt.figure("Data and model")
    plt.plot(center, hist, '.', label = "Data")
    plt.plot(xx, pdf_f, 'r', label = "Model, theta eye-fitted")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.plot(xx, w_normal_3, color = 'purple', label = "Norm 3", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()"""
    
    # try to initialise things
    
    two_class_model = FunctionalModel(counts, center, weighted_log_normal, bounds2, name2)
    three_class_model = FunctionalModel(counts, center, three_weighted_log_normal, bounds3, name3)

    run2  = False
    run3  = False
    nlive = 2000

    if run2 == True:
        work = raynest.raynest(two_class_model, verbose=2,                  # model on which infere, output on screen and memory (how much the function speaks)
                               nnest=1,                                     # parallelise: number of parallel algorithm
                               nensemble=3, nslice=0, nhamiltonian=0,       # method of replacing the live points
                               nlive=nlive, maxmcmc=5000,                    # number of live points (> 2*ndim + 1), max number of MCMC willing to take to have the new live point
                               resume=1, periodic_checkpoint_interval=100,  # resume True and save the situation of algorithm every periodic_time 
                               output='test')                   # where to save the results 
        work.run()

    if run3 == True:
        work = raynest.raynest(three_class_model, verbose=2,
                               nnest=1,
                               nensemble=3, nslice=0, nhamiltonian=0,
                               nlive=nlive, maxmcmc=5000,
                               resume=1, periodic_checkpoint_interval=100,
                               output='three_class_output')
        work.run()

    # take sample and evidence from the saved file
    import h5py
    filename = os.path.join("two_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples2 = h5_file['combined'].get('posterior_samples')
    evidence2 = h5_file['combined'].get('logZ')[()]
    
    filename = os.path.join("three_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples3 = h5_file['combined'].get('posterior_samples')
    evidence3 = h5_file['combined'].get('logZ')[()]

    print(f"The Odd ratio is O_23 = {evidence2/evidence3:.2e}")
    
    # plot the data with the best fit for 2 classes
    posterior_models = [weighted_log_normal(xx, s) for s in samples2]
    pdf2 = np.percentile(posterior_models,50,axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in samples2], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in samples2], 50, axis=0)
    
    plt.figure("Data and 2 class model")
    plt.plot(center, hist, '.', label = "Data")
    plt.plot(xx, pdf2, 'r', label = "Model")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()

    # plot the data with the best fit for 3 classes
    posterior_models = [three_weighted_log_normal(xx, s) for s in samples3]
    pdf3 = np.percentile(posterior_models,50,axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in samples3], 50, axis=0)
    w_normal_2 = np.percentile([s[7] * gauss(xx, s[3], s[4]) for s in samples3], 50, axis=0)
    w_normal_3 = np.percentile([(1-s[0]-s[7]) * gauss(xx, s[5], s[6]) for s in samples3], 50, axis=0)
    
    plt.figure("Data and 3 class model")
    plt.plot(center, hist, '.', label = "Data")
    plt.plot(xx, pdf3, 'r', label = "Model")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'purple', label = "Norm 2", alpha = 0.5)
    plt.plot(xx, w_normal_3, color = 'orange', label = "Norm 3", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()

    # plot the data with the two best fit
    plt.figure("Data and the two models")
    plt.plot(center, hist, '.', label = "Data")
    plt.plot(xx, pdf2, 'r', label = "Model with 2 classes")
    plt.plot(xx, pdf3, 'g', label = "Model with 3 classes")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    
    # given that I think they will be very similar plot the difference
    plt.figure("Difference of the two models")
    plt.plot(xx, pdf2-pdf3, 'C0', label = "Difference of the models (2-3)")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Difference")
    plt.legend()
    plt.show()