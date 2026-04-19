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
    counts, edges = np.histogram(log_T90, bins = bins)                      # used for the real analysis
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
    
    # Analyise the results
    # I'm going with Jeffreys priors
    filename = os.path.join("two_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples2 = h5_file['combined'].get('posterior_samples')
    log_evidence2 = h5_file['combined'].get('logZ')[()]
    
    filename = os.path.join("three_class_output","raynest.h5")
    h5_file = h5py.File(filename,'r')
    samples3 = h5_file['combined'].get('posterior_samples')
    log_evidence3 = h5_file['combined'].get('logZ')[()]

    print(f"The log evidence for 2 classes is: {log_evidence2:.2f}")
    print(f"The log evidence for 3 classes is: {log_evidence3:.2f}")
    print(f"The Odd ratio is O_23 = {np.exp(log_evidence2-log_evidence3):.2e}")
    
    # first of all: trace of the samples, burnin and thinning
    # change from array of tuples to array of arrays
    samples2 = samples2[()]
    sample = np.zeros((len(samples2),5))
    for i in range(len(name2)):
        sample[:,i] = samples2[name2[i]]
    samples2 = sample

    # trace plots
    burnin = 100
    fig = plt.figure("Parameters chain 2", figsize = (6,6))
    for i in range(samples2.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(samples2[:,i], '.', color = 'C0')
        ax.axvline(burnin, color='r', linestyle='dashed')
        ax.set_ylabel(name2_l[i])
    ax.set_xlabel("Step")
    plt.tight_layout()
    
    
    samples = samples2[burnin:,:]
    thinning = 15

    # look at autocorrelation for theta
    fig = plt.figure("Parameters autocorrelation 2", figsize = (6,6))
    for i in range(samples.shape[1]):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(autocorrelation(samples[:,i]), '.', color = 'C0', label = 'Autocorrelation')
        ax.axvline(thinning, color='r', linestyle='dashed')
    ax.set_xlabel("Step")
    plt.tight_layout()
    
    parameters = samples2[::thinning,:]

    print(f"2 class: after burn-in and thinning we have {len(parameters[:,0])} samples")

    # find peaks and credible interval
    par_val = np.zeros(len(name2))
    d_par_plus = np.zeros(len(name2))
    d_par_minus = np.zeros(len(name2))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)

    
    fig1 = plt.figure("Parameters histogram 2", figsize = (6,6))
    for i in range(parameters.shape[1]):
        if i > 0:
            ax = fig1.add_subplot(3, 2, i+2)
        else:
            ax = fig1.add_subplot(3, 2, i+1)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', linewidth = 1.5, baseline=0)        
        ax.axvline(par_val[i], color = 'green', linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', linestyle='dashed')
        ax.set_xlabel(name2_l[i])
        delta = 2*np.diff(bins_i)[0]
        ax.set_xlim(np.min(bins_i)-delta, np.max(bins_i)+delta)
        ax.set_ylim(0, np.max(counts_i) * 1.1)

    ax_leg = fig1.add_subplot(3,2,2)
    # Create dummy artists just for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='C0', linewidth=1.5, label='Marginalised posterior samples'),
        Line2D([0], [0], color='green', linestyle='dashed', label='Median value'),
        Line2D([0], [0], color='orange', linestyle='dashed', label='90% confidence interval')]
    ax_leg.legend(handles=legend_elements, loc='center')
    ax_leg.axis('off')  # Hide axes, ticks, spines and background
    plt.tight_layout()

    
    posterior_models = [weighted_log_normal(xx, s) for s in parameters]
    l, pdf2, h = np.percentile(posterior_models,[5,50,95],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)

    scale = np.diff(bins)[0]*np.sum(counts)
    # plot the data with the best fit
    fig2 = plt.figure("Data and model 2")
    ax = fig2.add_subplot(111)
    ax.stairs(counts, edges, color = 'C0', label = 'Data', linewidth=1.5, zorder =1)
    ax.plot(xx, pdf2*scale, 'r', label = "Model",zorder=4)
    ax.fill_between(xx, h*scale, l*scale, facecolor='salmon', alpha = 0.5, label="90% confidence", zorder =2)
    ax.plot(xx, w_normal_1*scale, 'g', label = "Norm 1", alpha = 0.75, zorder = 3)
    ax.plot(xx, w_normal_2*scale, color = 'darkorange', label = "Norm 2", alpha = 0.75, zorder = 3)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Counts")
    ax.set_ylim([0,170])
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    #store the parameters result with a different name to eventually save at end of file
    save2 = np.array([par_val, d_par_plus, d_par_minus]).T
    plt.close('all')

    # now all again for model with 3 class
    samples3 = samples3[()]
    sample = np.zeros((len(samples3),len(name3)))
    for i in range(len(name3)):
        sample[:,i] = samples3[name3[i]]
    samples3 = sample

    # trace plots
    burnin = 2000
    fig = plt.figure("Parameters chain 3", figsize = (6,6))
    for i in range(samples3.shape[1]):
        ax = fig.add_subplot(8, 1, i+1)
        ax.plot(samples3[:,i], '.', color = 'C0')
        ax.axvline(burnin, color='r', linestyle='dashed')
        ax.set_ylabel(name3_l[i])
    ax.set_xlabel("Step")
    plt.tight_layout()
        
    samples = samples3[burnin:,:]
    thinning = 6

    # look at autocorrelation for theta
    fig = plt.figure("Parameters autocorrelation 3", figsize = (6,6))
    for i in range(samples.shape[1]):
        ax = fig.add_subplot(8, 1, i+1)
        ax.plot(autocorrelation(samples[:,i]), '.-', color = 'C0', label = 'Autocorrelation')
        ax.axvline(thinning, color='r', linestyle='dashed')
        ax.set_xlim(-1, 20)
    ax.set_xlabel("Step")
    plt.tight_layout()

    parameters = samples[::thinning,:]
    print(f"3 class: after burn-in and thinning we have {len(parameters[:,0])} samples")


    # find peaks and credible interval
    par_val = np.zeros(len(name3))
    d_par_plus = np.zeros(len(name3))
    d_par_minus = np.zeros(len(name3))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)


    fig3 = plt.figure("Parameters histogram 3", figsize = (6,6))
    for i in range(parameters.shape[1]):
        if i == 7:
            ax = fig3.add_subplot(4, 2, 2)
        elif i == 0:
            ax = fig3.add_subplot(4, 2, 1)
        else:
            ax = fig3.add_subplot(4, 2, i+2)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', label = 'Posterior samples', linewidth = 1.5)
        """
        # plot the priors, Jeffrey for sigmas and uniform for others
        if i == 2 or i == 4 or i == 6:
            a = bounds3[i][0]
            b = bounds3[i][1]
            ss = np.linspace(a, b, 100)
            ax.plot(ss, 1/(np.log(b/a) * ss), color = 'r', label = "Prior", linestyle='dashed')
        else:
            ax.axhline(1/(bounds3[i][1] - bounds3[i][0]), color = 'r', label = "Prior", linestyle='dashed')
        """
        ax.axvline(par_val[i], color = 'g', label = "Median value", linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', label = "90% confidence interval", linestyle='dashed')
        ax.set_xlabel(name3_l[i])
        delta = 2*np.diff(bins_i)[0]
        ax.set_xlim(np.min(bins_i)-delta, np.max(bins_i)+delta)
        ax.set_ylim(0, np.max(counts_i) * 1.1)
    plt.tight_layout()

    
    posterior_models = [three_weighted_log_normal(xx, s) for s in parameters]
    l, pdf3, h = np.percentile(posterior_models,[5,50,95],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([s[7] * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)
    w_normal_3 = np.percentile([(1-s[0]-s[7]) * gauss(xx, s[5], s[6]) for s in parameters], 50, axis=0)

    # plot the data with the best fit
    scale = np.diff(bins)[0]*np.sum(counts)
    # plot the data with the best fit
    fig4 = plt.figure("Data and model 3")
    ax = fig4.add_subplot(111)
    ax.stairs(counts, edges, color = 'C0', label = 'Data', linewidth=1.5, zorder =1)
    ax.plot(xx, pdf3*scale, 'r', label = "Model",zorder=4)
    ax.fill_between(xx, h*scale, l*scale, facecolor='salmon', alpha = 0.5, label="90% confidence", zorder =2)
    ax.plot(xx, w_normal_1*scale, 'g', label = "Norm 1", alpha = 0.75, zorder = 3)
    ax.plot(xx, w_normal_2*scale, color = 'purple', label = "Norm 2", alpha = 0.75, zorder = 3)
    ax.plot(xx, w_normal_3*scale, color = 'darkorange', label = "Norm 3", alpha = 0.75, zorder = 3)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Counts")
    ax.set_ylim([0,170])
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()
    
    
    # plot the data with the two best fit
    fig5 = plt.figure("Data and the two models")
    ax = fig5.add_subplot(111)
    ax.stairs(counts, edges, color = 'C0', label = 'Data', linewidth=1.5, zorder =1)
    #ax.fill_between(xx, pdf2*scale, pdf3*scale, color = 'g', alpha = 0.5)
    ax.plot(xx, pdf2*scale, 'k', label = "Model with 2 classes", zorder = 3)
    ax.plot(xx, pdf3*scale, 'r', label = "Model with 3 classes", zorder = 2)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Counts")
    ax.set_ylim([0,170])
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend(loc='upper left')


    # save the results
    
    plt.show()
    exit()

    header = "par_val d_par_plus d_par_minus"
    # from the 2 class analysis
    fig1.savefig(main_dir+"\\Results\\4\\2_Parameters_histogram.png", dpi = 600)
    fig2.savefig(main_dir+"\\Results\\4\\2_Data_and_model.png", dpi = 600)
    np.savetxt(main_dir+"\\Results\\4\\2_parameters_values.txt", save2, header=header)

    # from the 3 class analysis
    fig3.savefig(main_dir+"\\Results\\4\\3_Parameters_histogram.png", dpi = 600)
    fig4.savefig(main_dir+"\\Results\\4\\3_Data_and_model.png", dpi = 600)
    np.savetxt(main_dir+"\\Results\\4\\3_parameters_values.txt", np.array([par_val, d_par_plus, d_par_minus]).T, header=header)

    # both models
    fig5.savefig(main_dir+"\\Results\\4\\Model_both.png", dpi = 600)