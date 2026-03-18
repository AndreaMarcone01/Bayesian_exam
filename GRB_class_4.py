# Python script to analyse the results of GRB_class_4_NS.py

import numpy as np
from scipy.special import xlogy

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import h5py

    rng = np.random.default_rng(111) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T90, d_log_T90, Hardness_R = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)
    
    nbins = 50
    bins = np.linspace(-4, 7, nbins)
    hist, edges = np.histogram(log_T90, bins = bins, density = True)    # used in plot
    counts, _ = np.histogram(log_T90, bins = bins)                      # used for the real analysis
    center = (edges[1:] + edges[:-1])*0.5

    name2 = ["$w$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$"]
    bounds2 = np.array([[0.0,1.0], [-4.0,7.0], [0.01,3.0] , [-4.0, 7.0], [0.01,3.0]])

    name3 = ["$w_1$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$", "$\\mu_3$", "$\\sigma_3$", "$w_2$"]
    # updated bounds given the precedent results
    bounds3 = np.array([[0.0,1.0], 
                        [-4.0, 2.0], [0.01,3.0],     # first gaussian on the short GRB
                        [-4.0, 7.0], [0.01,3.0],     # intermediate? Less known
                        [0.0, 7.0], [0.01,3.0],      # third on long GRB
                        [0.0,1.0]])
    
    xx = np.linspace(-4,7,100)
    
    # take sample and evidence from the saved file
    # first for the case of uniform priors, then Jeffereys
    
    filename = os.path.join("two_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples2 = h5_file['combined'].get('posterior_samples')
    log_evidence2 = h5_file['combined'].get('logZ')[()]
    
    filename = os.path.join("three_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples3 = h5_file['combined'].get('posterior_samples')
    log_evidence3 = h5_file['combined'].get('logZ')[()]

    print(f"The Odd ratio is O_23 = {np.exp(log_evidence2-log_evidence3):.2e} (uniform priors)")

    # first of all: trace of the samples, burnin and thinning
    print(
        np.vstack([samples2[()][t] for t in range(len(samples2))])
                    )

    exit()
    # plot the data with the best fit for 2 classes
    posterior_models = [weighted_log_normal(xx, s) for s in samples2]
    pdf2 = np.percentile(posterior_models,50,axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in samples2], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in samples2], 50, axis=0)
    
    plt.figure("Data and 2 class model", figsize = (6.4, 4.8))
    plt.stairs(hist, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    plt.plot(xx, pdf2, 'r', label = "Model")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\Model_2.png", dpi = 600)

    # plot the data with the best fit for 3 classes
    posterior_models = [three_weighted_log_normal(xx, s) for s in samples3]
    pdf3 = np.percentile(posterior_models,50,axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in samples3], 50, axis=0)
    w_normal_2 = np.percentile([s[7] * gauss(xx, s[3], s[4]) for s in samples3], 50, axis=0)
    w_normal_3 = np.percentile([(1-s[0]-s[7]) * gauss(xx, s[5], s[6]) for s in samples3], 50, axis=0)
    
    plt.figure("Data and 3 class model", figsize = (6.4, 4.8))
    plt.stairs(hist, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    plt.plot(xx, pdf3, 'r', label = "Model")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'purple', label = "Norm 2", alpha = 0.5)
    plt.plot(xx, w_normal_3, color = 'orange', label = "Norm 3", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\Model_3.png", dpi = 600)

    # plot the data with the two best fit
    plt.figure("Data and the two models", figsize = (6.4, 4.8))
    plt.stairs(hist, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    plt.plot(xx, pdf2, 'r', label = "Model with 2 classes")
    plt.plot(xx, pdf3, 'g', label = "Model with 3 classes")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\Model_both.png", dpi = 600)
    
    # given that I think they will be very similar plot the difference
    plt.figure("Difference of the two models", figsize = (6.4, 4.8))
    plt.plot(xx, pdf2-pdf3, 'C0', label = "Difference of the models (2-3)")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Difference")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\Model_difference.png", dpi = 600)
    #plt.show()

    #Jeffreys priors

    filename = os.path.join("J_two_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples2 = h5_file['combined'].get('posterior_samples')
    log_evidence2 = h5_file['combined'].get('logZ')[()]
    
    filename = os.path.join("J_three_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples3 = h5_file['combined'].get('posterior_samples')
    log_evidence3 = h5_file['combined'].get('logZ')[()]

    print(f"The Odd ratio is O_23 = {np.exp(log_evidence2-log_evidence3):.2e}")
    
    # plot the data with the best fit for 2 classes
    posterior_models = [weighted_log_normal(xx, s) for s in samples2]
    pdf2 = np.percentile(posterior_models,50,axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in samples2], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in samples2], 50, axis=0)
    
    plt.figure("Data and 2 class model", figsize = (6.4, 4.8))
    plt.stairs(hist, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    plt.plot(xx, pdf2, 'r', label = "Model")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\J_Model_2.png", dpi = 600)

    # plot the data with the best fit for 3 classes
    posterior_models = [three_weighted_log_normal(xx, s) for s in samples3]
    pdf3 = np.percentile(posterior_models,50,axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in samples3], 50, axis=0)
    w_normal_2 = np.percentile([s[7] * gauss(xx, s[3], s[4]) for s in samples3], 50, axis=0)
    w_normal_3 = np.percentile([(1-s[0]-s[7]) * gauss(xx, s[5], s[6]) for s in samples3], 50, axis=0)
    
    plt.figure("Data and 3 class model", figsize = (6.4, 4.8))
    plt.stairs(hist, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    plt.plot(xx, pdf3, 'r', label = "Model")
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'purple', label = "Norm 2", alpha = 0.5)
    plt.plot(xx, w_normal_3, color = 'orange', label = "Norm 3", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\J_Model_3.png", dpi = 600)

    # plot the data with the two best fit
    plt.figure("Data and the two models", figsize = (6.4, 4.8))
    plt.stairs(hist, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    plt.plot(xx, pdf2, 'r', label = "Model with 2 classes")
    plt.plot(xx, pdf3, 'g', label = "Model with 3 classes")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\J_Model_both.png", dpi = 600)
    
    # given that I think they will be very similar plot the difference
    plt.figure("Difference of the two models", figsize = (6.4, 4.8))
    plt.plot(xx, pdf2-pdf3, 'C0', label = "Difference of the models (2-3)")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Difference")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\J_Model_difference.png", dpi = 600)
    #plt.show()