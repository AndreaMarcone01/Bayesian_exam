# Python script for point 1b: find parameters of the classification model with uncertainties on the measures

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

def log_likelihood(theta, data, err):
    """Log likelihood for model with errors. Convolution between the two gaussians of the model and the gaussian of the error
    
    Args:
        theta (array): parameters of the model
        data (array): measured points
        err (array): gaussian errors associated to data
        
    Returns:
        likelihood (float): log of the likelihood
    """

    w = theta[0]
    mu_1 = theta[1]
    sigma_1 = theta[2]
    mu_2 = theta[3]
    sigma_2 = theta[4]

    conv_1 = w * gauss(data, mu_1, np.sqrt(sigma_1**2 + err**2))
    conv_2 = (1-w) * gauss(data, mu_2, np.sqrt(sigma_2**2 + err**2))
    log_like = np.log(conv_1 + conv_2)
    return np.sum(log_like)

def log_posterior(theta, data, err, bounds):
    """Log posterior for model.
    
    Args:
        theta (array): parameters of the model
        data (array): measured points
        err (array): gaussian errors associated to data
        bounds (array): bound for each parameter, expected as [min, max]
        
    Returns:
        posterior (array): log of the posterior
    """
    
    posterior = log_prior(theta, bounds) + log_likelihood(theta, data, err)
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
        fname = main_dir+"\\unc_samples_covariance.txt"
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

def metropolis_hastings(theta0, postpdf, data, err, bounds, rng, blind, n = 10000):
    """Metropolis hastings algorithm to sample the posterior.
    
    Args:
        theta0 (array): initial point for the chain
        postpdf (function): posterior to use for the chain
        data (array): measured points
        err (array): gaussian errors associated to data
        bounds (array): bound for each parameter, expected as [min, max]
        rng (np.random.default_rng): default rng for reproduce results
        blind (boolean): decide if run with a blind covariance or use a already note covariance
        n (float): length of the chain, default to 10000
    
    Returns:
        samples (array): samples of the parameters
    """

    d = theta0.shape[0]
    logP0 = postpdf(theta0, data, err, bounds)
    samples = np.zeros((n,d), dtype=np.float64)
    logP = np.zeros(n, dtype=np.float64)
    accepted = 0
    rejected = 0

    for i in range(n):
        theta_try = theta0 + proposed_distribution(theta0, bounds, rng, blind)
        logP_try = postpdf(theta_try, data, err, bounds)
        
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
    log_T, d_log_T, Hardness_R = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)
    
    nbins = 50
    bins = np.linspace(-4, 7, nbins)
    counts, edges = np.histogram(log_T, bins = bins)                      # used for the real analysis
    center = (edges[1:] + edges[:-1])*0.5

    par_name = np.array(["$w$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$"])
    bounds = np.array([[0.0,1.0], [-4.0,7.0], [0.01,3.0] , [-4.0, 7.0], [0.01,3.0]])
    theta_0 = np.array([rng.uniform(0,1),
                        rng.uniform(-4,7), 
                        rng.uniform(0.01,3),
                        rng.uniform(-4,7),
                        rng.uniform(0.01,3)])
    
    xx = np.linspace(-4,7,256)
    dx = np.diff(xx)[0]

    # build data with uncertainties
    data = np.zeros(len(xx))
    for i in range(len(log_T)):
        data = data + gauss(xx, log_T[i], d_log_T[i]) * dx

    data_n = data/(np.sum(data) * dx)               # normalize to have area 1 (for plots)
    scale = np.sum(counts)*np.diff(bins)[0]         # scale from normalized data to binned ones
    
    # plot the data
    fig0 = plt.figure("Data")
    ax = fig0.add_subplot(111)
    ax.plot(xx, data_n*scale, 'r', alpha = 0.6, label = "Data with uncertainties")
    ax.stairs(counts, edges, color = 'C0', label = 'Data', linewidth = 1.5)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Counts")
    ax.set_ylim([0.0, 170])
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()


    # try to initialise things
    run = False
    prep_run = True

    if run == True:
        samples, logP = metropolis_hastings(theta_0, log_posterior, log_T, d_log_T, 
                                            bounds, rng, blind = True)
        
        # use this first estimate to adjust the proposal
        covariance = np.cov(samples.T)
        np.savetxt(main_dir+"\\unc_samples_covariance.txt", covariance)

        if prep_run == True:
            # re-run chain with new covariance in proposal
            samples, logP = metropolis_hastings(theta_0, log_posterior, log_T, d_log_T,
                                                bounds, rng, blind = False, n=100000)
        
        # save the chain
        header = "w , mu_1, sigma_1, mu_2, sigma_2"
        np.savetxt(main_dir+"\\unc_samples_chain.txt", samples, header=header)
        np.savetxt(main_dir+"\\unc_samples_posterior.txt", logP, header="log posterior")

    else: 
        samples = np.loadtxt(main_dir+"\\unc_samples_chain.txt")
        logP = np.loadtxt(main_dir+"\\unc_samples_posterior.txt")
        

    # Look at the results
    fig1 = plt.figure("Parameters chain", figsize = (6,6))
    for i in range(samples.shape[1]):
        ax = fig1.add_subplot(5, 1, i+1)
        ax.plot(samples[:,i], '.', color = 'C0')
        ax.set_ylabel(par_name[i])
        ax.grid(linestyle = 'dashed')
        ax.set_axisbelow(True)
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    burnin = 300 
    fig2 = plt.figure("Parameters chain: zoom to burn-in", figsize = (6,6))
    for i in range(samples.shape[1]):
        ax = fig2.add_subplot(5, 1, i+1)
        ax.plot(samples[:,i], '.', color = 'C0')
        ax.axvline(burnin, color = 'r', label = 'Burn-in', linestyle = 'dashed')
        ax.set_xlim(-10, 2*burnin)
        ax.set_ylabel(par_name[i])
        ax.grid(linestyle = 'dashed')
        ax.set_axisbelow(True)
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    # look at the posterior 
    fig_post0 = plt.figure("Chain posterior")
    ax = fig_post0.add_subplot(111)
    ax.plot(logP, label = 'log Posterior')
    ax.set_ylabel("log Posterior")
    ax.set_xlabel("Iteration")
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    fig_post = plt.figure("Chain posterior zoom")
    ax = fig_post.add_subplot(111)
    ax.plot(logP, label = 'log Posterior')
    ax.axvline(burnin, color = 'r', label = 'Burn-in', linestyle = 'dashed')
    ax.set_ylabel("log Posterior")
    ax.set_xlabel("Iteration")
    ax.set_xlim(-10, 2*burnin)
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()


    # look at autocorrelation for theta
    fig3 = plt.figure("Parameters autocorrelation", figsize = (6,6))
    for i in range(samples.shape[1]):
        ax = fig3.add_subplot(5, 1, i+1)
        ax.plot(autocorrelation(samples[:,i]), '.', color = 'C0', label = 'Autocorrelation')
        ax.set_ylabel(par_name[i])
        ax.grid(linestyle = 'dashed')
        ax.set_axisbelow(True)
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    thinning = 200
    fig4 = plt.figure("Parameters autocorrelation: zoom to thinning", figsize = (6,6))
    for i in range(samples.shape[1]):
        ax = fig4.add_subplot(5, 1, i+1)
        autoc = autocorrelation(samples[:,i])
        ax.plot(autoc, '.', color = 'C0', label = f'Value at thinning: {autoc[thinning]:.2f}')
        ax.axvline(thinning, color = 'r', linestyle = 'dashed')
        ax.set_ylabel(par_name[i])
        ax.set_xlim(-10, 2 * thinning)
        ax.legend()
        ax.grid(linestyle = 'dashed')
        ax.set_axisbelow(True)
    ax.set_xlabel("Iteration")
    plt.tight_layout()
    
    # after burn-in and autocorrelation we plot the histograms of the parameters
    parameters = samples[burnin:,:]
    parameters = parameters[::thinning,:]

    print(f"After burn-in and thinning we have {len(parameters[:,0])} samples")
    
    # find peaks and credible interval
    par_val = np.zeros(len(par_name))
    d_par_plus = np.zeros(len(par_name))
    d_par_minus = np.zeros(len(par_name))
    for i in range(parameters.shape[1]):
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        center_i = 0.5*(bins_i[1:] + bins_i[:-1])
        par_val[i], d_par_plus[i], d_par_minus[i] = errors_around_peak(center_i, counts_i)

    
    fig5 = plt.figure("Parameters histogram", figsize = (6,6))
    for i in range(parameters.shape[1]):
        if i > 0:
            ax = fig5.add_subplot(3, 2, i+2)
        else:
            ax = fig5.add_subplot(3, 2, i+1)
        counts_i, bins_i = np.histogram(parameters[:,i], bins = 30, density=True)
        ax.stairs(counts_i, bins_i, color = 'C0', linewidth = 1.5, baseline=0)
        """
        # plot the priors, Jeffrey for sigmas and uniform for others
        if i == 2 or i == 4:
            a = bounds[i][0]
            b = bounds[i][1]
            ss = np.linspace(a, b, 100)
            ax.plot(ss, 1/(np.log(b/a) * ss), color = 'r', linestyle='dashed')
        else:
            ax.axhline(1/(bounds[i][1] - bounds[i][0]), color = 'r', linestyle='dashed')
        """
        ax.axvline(par_val[i], color = 'green', linestyle='dashed')
        ax.axvline(par_val[i]+d_par_plus[i], color = 'orange', linestyle='dashed')
        ax.axvline(par_val[i]-d_par_minus[i], color = 'orange', linestyle='dashed')
        ax.set_xlabel(par_name[i])
        delta = 2*np.diff(bins_i)[0]
        ax.set_xlim(np.min(bins_i)-delta, np.max(bins_i)+delta)
        ax.set_ylim(0, np.max(counts_i) * 1.1)

    ax_leg = fig5.add_subplot(3,2,2)
    # Create dummy artists just for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='C0', linewidth=1.5, label='Marginalized posterior samples'),
        #Line2D([0], [0], color='r', linestyle='dashed', label='Prior'),
        Line2D([0], [0], color='green', linestyle='dashed', label='Median value'),
        Line2D([0], [0], color='orange', linestyle='dashed', label='Median value $\\pm\\sigma$')]
    ax_leg.legend(handles=legend_elements, loc='center')
    ax_leg.axis('off')  # Hide axes, ticks, spines and background
    plt.tight_layout()

    posterior_models = [weighted_log_normal(xx, s) for s in parameters]
    l, pdf, h = np.percentile(posterior_models,[5,50,95],axis=0)
    w_normal_1 = np.percentile([s[0] * gauss(xx, s[1], s[2]) for s in parameters], 50, axis=0)
    w_normal_2 = np.percentile([(1-s[0]) * gauss(xx, s[3], s[4]) for s in parameters], 50, axis=0)

    scale = np.sum(counts)*np.diff(bins)[0]
    # plot the data with the best fit
    fig6 = plt.figure("Data and model")
    ax = fig6.add_subplot(111)
    ax.plot(xx, data_n*scale, color = 'C0', label = 'Data', zorder=1)
    ax.plot(xx, pdf*scale, 'r', label = "Model", zorder = 4)
    ax.fill_between(xx, h*scale, l*scale, facecolor='salmon', alpha = 0.5, label="90% confidence", zorder=2)
    ax.plot(xx, w_normal_1*scale, 'g', label = "Norm 1", alpha = 0.75,zorder=3)
    ax.plot(xx, w_normal_2*scale, color = 'darkorange', label = "Norm 2", alpha = 0.75,zorder=3)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Counts")
    ax.set_ylim([0,170])
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    # end of  point: save, show or close all the open figures
    
    #plt.show()
    #exit()

    fig0.savefig(main_dir+"\\Results\\1b\\Err_Dataset.png", dpi = 600)
    fig1.savefig(main_dir+"\\Results\\1b\\Err_Parameters_chain.png", dpi = 600)
    fig2.savefig(main_dir+"\\Results\\1b\\Err_Parameters_chain_zoom.png", dpi = 600)
    fig_post0.savefig(main_dir+"\\Results\\1b\\Err_Posterior_chain.png", dpi = 600)
    fig_post.savefig(main_dir+"\\Results\\1b\\Err_Posterior_chain_zoom.png", dpi = 600)
    fig3.savefig(main_dir+"\\Results\\1b\\Err_Parameters_autocorr.png", dpi = 600)
    fig4.savefig(main_dir+"\\Results\\1b\\Err_Parameters_autocorr_zoom.png", dpi = 600)
    fig5.savefig(main_dir+"\\Results\\1b\\Err_Parameters_hist.png", dpi = 600)
    fig6.savefig(main_dir+"\\Results\\1b\\Err_Data_and_model.png", dpi = 600)
    
    header = "par_val d_par_plus d_par_minus"
    np.savetxt(main_dir+"\\Results\\1b\\Err_parameters_values.txt", np.array([par_val, d_par_plus, d_par_minus]).T, header=header)

    header = "model w_normal_1 w_normal_2"
    np.savetxt(main_dir+"\\Results\\1b\\Err_model_values.txt", np.array([pdf, w_normal_1, w_normal_2]).T, header=header)