# Python script to analyse the results of GRB_class_4_NS.py

import numpy as np
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
    import h5py

    rng = np.random.default_rng(111) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T90, d_log_T90, Hardness_R = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)
    
    nbins = 50
    bins = np.linspace(-4, 7, nbins)
    hist, edges = np.histogram(log_T90, bins = bins, density = True)    # used in plot
    counts, _ = np.histogram(log_T90, bins = bins)                      # used for the real analysis
    center = (edges[1:] + edges[:-1])*0.5

    name2_l = ["$w$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$"]
    name2 = ["w", "mu_1", "sigma_1", "mu_2", "sigma_2"]
    bounds2 = np.array([[0.0,1.0], [-4.0,7.0], [0.01,3.0] , [-4.0, 7.0], [0.01,3.0]])

    name3_l = ["$w_1$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$", "$\\mu_3$", "$\\sigma_3$", "$w_2$"]
    name3 = ["w_1", "mu_1", "sigma_1", "mu_2", "sigma_2", "mu_3", "sigma_3", "w_2"]
    # updated bounds given the precedent results
    bounds3 = np.array([[0.0,1.0], 
                        [-4.0, 2.0], [0.01,3.0],     # first gaussian on the short GRB
                        [-4.0, 7.0], [0.01,3.0],     # intermediate? Less known
                        [0.0, 7.0], [0.01,3.0],      # third on long GRB
                        [0.0,1.0]])
    
    xx = np.linspace(-4,7,256)
    
    # take sample and evidence from the saved file
    """
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
    # change from array of tuples to array of arrays
    samples2 = samples2[()]
    sample = np.zeros((len(samples2),5))
    for i in range(len(name2)):
        sample[:,i] = samples2[name2[i]]
    samples2 = sample

    # trace plots
    fig = plt.figure("Parameters chain 2", figsize = (6,6))
    for i in range(samples2.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(samples2[:,i], '.', color = 'C0')
        ax.set_ylabel(name2_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    # honestly burn-in = 0

    # look at autocorrelation for theta
    fig = plt.figure("Parameters autocorrelation 2", figsize = (6,6))
    for i in range(samples2.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(autocorrelation(samples2[:,i]), '.', color = 'C0', label = 'Autocorrelation')
        ax.set_ylabel(name2_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    thinning = 10
    parameters = samples2[::thinning,:]

    # find peaks and credible interval
    par_val = np.zeros(len(name2))
    d_par_plus = np.zeros(len(name2))
    d_par_minus = np.zeros(len(name2))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)

    header = "par_val d_par_plus d_par_minus"
    np.savetxt(main_dir+"\\Results\\NS\\2_parameters_values.txt", np.array([par_val, d_par_plus, d_par_minus]).T, header=header)


    fig = plt.figure("Parameters histogram 2", figsize = (6,6))
    for i in range(parameters.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', label = 'Posterior samples', linewidth = 1.5)
        ax.axhline(1/(bounds2[i][1] - bounds2[i][0]), color = 'r', label = "Prior", linestyle='dashed')

        ax.axvline(par_val[i], color = 'g', label = "Peak value", linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', label = "Peak value $\\pm\\sigma$", linestyle='dashed')
        ax.set_xlabel(name2_l[i])
        ax.set_xlim(np.min(bins_i)*0.9, np.max(bins_i)*1.1)
        ax.set_ylim(0, np.max(counts_i) * 1.1)
    plt.tight_layout()
    plt.savefig(main_dir+"\\Results\\NS\\2_Parameters_histogram.png", dpi = 600)


    posterior_models = [weighted_log_normal(xx, s) for s in parameters]
    l, pdf2, h = np.percentile(posterior_models,[16,50,84],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)

    # plot the data with the best fit
    plt.figure("Data and model 2")
    plt.stairs(hist, edges, color = 'C0', label = 'Data')
    plt.plot(xx, pdf2, 'r', label = "Model")
    plt.fill_between(xx, h, l, facecolor='tomato', alpha = 0.5)
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(linestyle = 'dashed')
    plt.savefig(main_dir+"\\Results\\NS\\2_Data_and_model.png", dpi = 600)

    # now all again for model with 3 class
    samples3 = samples3[()]
    sample = np.zeros((len(samples3),len(name3)))
    for i in range(len(name3)):
        sample[:,i] = samples3[name3[i]]
    samples3 = sample

    # trace plots
    fig = plt.figure("Parameters chain 3", figsize = (6,6))
    for i in range(samples3.shape[1]):
        ax = fig.add_subplot(8, 1, i+1)
        ax.plot(samples3[:,i], '.', color = 'C0')
        ax.set_ylabel(name3_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    
    burnin = 1000

    # look at autocorrelation for theta
    fig = plt.figure("Parameters autocorrelation 3", figsize = (6,6))
    for i in range(samples3.shape[1]):
        ax = fig.add_subplot(8, 1, i+1)
        ax.plot(autocorrelation(samples3[:,i]), '.', color = 'C0', label = 'Autocorrelation')
        ax.set_ylabel(name3_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    thinning = 10
    sample = samples3[burnin:,:]
    parameters = sample[::thinning,:]

    # find peaks and credible interval
    par_val = np.zeros(len(name3))
    d_par_plus = np.zeros(len(name3))
    d_par_minus = np.zeros(len(name3))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)

    header = "par_val d_par_plus d_par_minus"
    np.savetxt(main_dir+"\\Results\\NS\\3_parameters_values.txt", np.array([par_val, d_par_plus, d_par_minus]).T, header=header)


    fig = plt.figure("Parameters histogram 3", figsize = (6,6))
    for i in range(parameters.shape[1]):
        if i == 7:
            ax = fig.add_subplot(4, 2, 2)
        elif i == 0:
            ax = fig.add_subplot(4, 2, 1)
        else:
            ax = fig.add_subplot(4, 2, i+2)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', label = 'Posterior samples', linewidth = 1.5)
        ax.axhline(1/(bounds3[i][1] - bounds3[i][0]), color = 'r', label = "Prior", linestyle='dashed')
        ax.axvline(par_val[i], color = 'g', label = "Median value", linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', label = "Peak value $\\pm\\sigma$", linestyle='dashed')
        ax.set_xlabel(name3_l[i])
        ax.set_xlim(np.min(bins_i)*0.9, np.max(bins_i)*1.1)
        ax.set_ylim(0, np.max(counts_i) * 1.1)
    plt.tight_layout()
    plt.savefig(main_dir+"\\Results\\NS\\3_Parameters_histogram.png", dpi = 600)

    posterior_models = [three_weighted_log_normal(xx, s) for s in parameters]
    l, pdf3, h = np.percentile(posterior_models,[16,50,84],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([s[7] * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)
    w_normal_3 = np.percentile([(1-s[0]-s[7]) * gauss(xx, s[5], s[6]) for s in parameters], 50, axis=0)

    # plot the data with the best fit
    plt.figure("Data and model 3")
    plt.stairs(hist, edges, color = 'C0', label = 'Data')
    plt.plot(xx, pdf3, 'r', label = "Model")
    plt.fill_between(xx, h, l, facecolor='tomato', alpha = 0.5)
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, 'purple', label = "Norm 2", alpha = 0.5)
    plt.plot(xx, w_normal_3, color = 'orange', label = "Norm 3", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(linestyle = 'dashed')
    plt.savefig(main_dir+"\\Results\\NS\\3_Data_and_model.png", dpi = 600)
    
    # plot the data with the two best fit
    plt.figure("Data and the two models")
    plt.stairs(hist, edges, color = 'C0', label = 'Data')
    plt.plot(xx, pdf2, 'k', label = "Model with 2 classes")
    plt.plot(xx, pdf3, 'r', label = "Model with 3 classes")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(linestyle = 'dashed')
    plt.savefig(main_dir+"\\Results\\NS\\Model_both.png", dpi = 600)

    #plt.show()
    #exit()

    plt.close('all')
    """

    #I'm going with Jeffreys priors

    filename = os.path.join("J_two_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples2 = h5_file['combined'].get('posterior_samples')
    log_evidence2 = h5_file['combined'].get('logZ')[()]
    
    filename = os.path.join("J_three_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples3 = h5_file['combined'].get('posterior_samples')
    log_evidence3 = h5_file['combined'].get('logZ')[()]

    print(f"The Odd ratio is O_23 = {np.exp(log_evidence2-log_evidence3):.2e}")
    
    # first of all: trace of the samples, burnin and thinning
    # change from array of tuples to array of arrays
    samples2 = samples2[()]
    sample = np.zeros((len(samples2),5))
    for i in range(len(name2)):
        sample[:,i] = samples2[name2[i]]
    samples2 = sample

    # trace plots
    fig = plt.figure("J Parameters chain 2", figsize = (6,6))
    for i in range(samples2.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(samples2[:,i], '.', color = 'C0')
        ax.set_ylabel(name2_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    # honestly burn-in = 0

    # look at autocorrelation for theta
    fig = plt.figure("J Parameters autocorrelation 2", figsize = (6,6))
    for i in range(samples2.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(autocorrelation(samples2[:,i]), '.', color = 'C0', label = 'Autocorrelation')
        ax.set_ylabel(name2_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    thinning = 10
    parameters = samples2[::thinning,:]

    # find peaks and credible interval
    par_val = np.zeros(len(name2))
    d_par_plus = np.zeros(len(name2))
    d_par_minus = np.zeros(len(name2))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)

    header = "par_val d_par_plus d_par_minus"
    np.savetxt(main_dir+"\\Results\\NS\\J_2_parameters_values.txt", np.array([par_val, d_par_plus, d_par_minus]).T, header=header)


    fig = plt.figure("J Parameters histogram 2", figsize = (6,6))
    for i in range(parameters.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', label = 'Posterior samples', linewidth = 1.5)
        # plot the priors, Jeffrey for sigmas and uniform for others
        if i == 2 or i == 4 or i == 6:
            a = bounds2[i][0]
            b = bounds2[i][1]
            ss = np.linspace(a, b, 100)
            ax.plot(ss, 1/(np.log(b/a) * ss), color = 'r', label = "Prior", linestyle='dashed')
        else:
            ax.axhline(1/(bounds2[i][1] - bounds2[i][0]), color = 'r', label = "Prior", linestyle='dashed')
        ax.axvline(par_val[i], color = 'g', label = "Peak value", linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', label = "Peak value $\\pm\\sigma$", linestyle='dashed')
        ax.set_xlabel(name2_l[i])
        ax.set_xlim(np.min(bins_i)*0.9, np.max(bins_i)*1.1)
        ax.set_ylim(0, np.max(counts_i) * 1.1)
    plt.tight_layout()
    plt.savefig(main_dir+"\\Results\\NS\\J_2_Parameters_histogram.png", dpi = 600)


    posterior_models = [weighted_log_normal(xx, s) for s in parameters]
    l, pdf2, h = np.percentile(posterior_models,[16,50,84],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)

    # plot the data with the best fit
    plt.figure("J Data and model 2")
    plt.stairs(hist, edges, color = 'C0', label = 'Data')
    plt.plot(xx, pdf2, 'r', label = "Model")
    plt.fill_between(xx, h, l, facecolor='tomato', alpha = 0.5)
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(linestyle = 'dashed')
    plt.savefig(main_dir+"\\Results\\NS\\J_2_Data_and_model.png", dpi = 600)

    # now all again for model with 3 class
    samples3 = samples3[()]
    sample = np.zeros((len(samples3),len(name3)))
    for i in range(len(name3)):
        sample[:,i] = samples3[name3[i]]
    samples3 = sample

    # trace plots
    fig = plt.figure("J Parameters chain 3", figsize = (6,6))
    for i in range(samples3.shape[1]):
        ax = fig.add_subplot(8, 1, i+1)
        ax.plot(samples3[:,i], '.', color = 'C0')
        ax.set_ylabel(name3_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    
    burnin = 1000

    # look at autocorrelation for theta
    fig = plt.figure("J Parameters autocorrelation 3", figsize = (6,6))
    for i in range(samples3.shape[1]):
        ax = fig.add_subplot(8, 1, i+1)
        ax.plot(autocorrelation(samples3[:,i]), '.', color = 'C0', label = 'Autocorrelation')
        ax.set_ylabel(name3_l[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    thinning = 10
    sample = samples3[burnin:,:]
    parameters = sample[::thinning,:]

    # find peaks and credible interval
    par_val = np.zeros(len(name3))
    d_par_plus = np.zeros(len(name3))
    d_par_minus = np.zeros(len(name3))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)

    header = "par_val d_par_plus d_par_minus"
    np.savetxt(main_dir+"\\Results\\NS\\J_3_parameters_values.txt", np.array([par_val, d_par_plus, d_par_minus]).T, header=header)


    fig = plt.figure("J Parameters histogram 3", figsize = (6,6))
    for i in range(parameters.shape[1]):
        if i == 7:
            ax = fig.add_subplot(4, 2, 2)
        elif i == 0:
            ax = fig.add_subplot(4, 2, 1)
        else:
            ax = fig.add_subplot(4, 2, i+2)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', label = 'Posterior samples', linewidth = 1.5)
        # plot the priors, Jeffrey for sigmas and uniform for others
        if i == 2 or i == 4 or i == 6:
            a = bounds3[i][0]
            b = bounds3[i][1]
            ss = np.linspace(a, b, 100)
            ax.plot(ss, 1/(np.log(b/a) * ss), color = 'r', label = "Prior", linestyle='dashed')
        else:
            ax.axhline(1/(bounds3[i][1] - bounds3[i][0]), color = 'r', label = "Prior", linestyle='dashed')
        ax.axvline(par_val[i], color = 'g', label = "Median value", linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', label = "Peak value $\\pm\\sigma$", linestyle='dashed')
        ax.set_xlabel(name3_l[i])
        ax.set_xlim(np.min(bins_i)*0.9, np.max(bins_i)*1.1)
        ax.set_ylim(0, np.max(counts_i) * 1.1)
    plt.tight_layout()
    plt.savefig(main_dir+"\\Results\\NS\\J_3_Parameters_histogram.png", dpi = 600)


    posterior_models = [three_weighted_log_normal(xx, s) for s in parameters]
    l, pdf3, h = np.percentile(posterior_models,[16,50,84],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([s[7] * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)
    w_normal_3 = np.percentile([(1-s[0]-s[7]) * gauss(xx, s[5], s[6]) for s in parameters], 50, axis=0)

    # plot the data with the best fit
    plt.figure("J Data and model 3")
    plt.stairs(hist, edges, color = 'C0', label = 'Data')
    plt.plot(xx, pdf3, 'r', label = "Model")
    plt.fill_between(xx, h, l, facecolor='tomato', alpha = 0.5)
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, 'purple', label = "Norm 2", alpha = 0.5)
    plt.plot(xx, w_normal_3, color = 'orange', label = "Norm 3", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(linestyle = 'dashed')
    plt.savefig(main_dir+"\\Results\\NS\\J_3_Data_and_model.png", dpi = 600)
    
    # plot the data with the two best fit
    plt.figure("J Data and the two models")
    plt.stairs(hist, edges, color = 'C0', label = 'Data')
    plt.plot(xx, pdf2, 'k', label = "Model with 2 classes")
    plt.plot(xx, pdf3, 'r', label = "Model with 3 classes")
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(linestyle = 'dashed')
    plt.savefig(main_dir+"\\Results\\NS\\J_Model_both.png", dpi = 600)

    #plt.show()