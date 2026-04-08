# Python script for point 5: 2D classification with hardness ratio

import numpy as np
import math
from pdf_analysis import errors_around_peak
from scipy.stats import multivariate_normal, multivariate_t, invwishart

def gauss(x, mu, sigma):
    """A gaussian.
    
    Args:
        x (array): normal distributed variable, this is expected as log
        mu (float): mean of the distribution
        sigma (float): variance of the distribution
        
    Returns:
        pdf (array): probability density function, normalized to 1
    """

    pdf = 1/(np.sqrt(2 * np.pi)*sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    return pdf

class state():
    def __init__(self, N_cluster, data, mu_0, alpha, rng):
        # compute some suff_stats that can't be done in a line
        # random initial assignment
        vec_z = rng.choice(N_cluster, size=data.shape[0])
        identity = np.diag(np.full(data.shape[1],1))
                
        # given the assignments first bar_x_k
        bar_x_k = np.zeros((N_cluster, data.shape[1]))      # first index 0=log_T, 1=log_HR, second index = k
        for k in range(N_cluster):
            bar_x_k[k] = np.mean(data[vec_z == k], axis = 0)

        # check that the dimentions are right
        if bar_x_k.shape != mu_0.shape:
            print("Something is wrong, check the initial mu_0!")
            print("Exiting...")
            exit()

        # initial scatter matrix
        S_k = np.zeros((N_cluster, data.shape[1], data.shape[1]))
        for k in range(N_cluster):
            data_k = data[vec_z==k]
            diff = data_k - data_k.mean(axis=0)
            S_k[k] =  diff.T @ diff 

        # dictionary with important stuff. Trailing underscore means that the quantity is constant
        self.state = {
            "N_cluster_": N_cluster,                                    # number of cluster (=3)
            "cluster_id_": range(N_cluster),                            # id of cluster [0,1,2]
            "data_" : data,                                             # data to assign
            "N_data_": len(data),                                       # number of points
            "hyperparameters_": {
                "mu_0": mu_0,                                           # mu_0 for every cluster
                "Psi_0": np.full(S_k.shape, identity),                  # covariance_0 for every cluster
                "k_0": np.full(N_cluster, 0.1),                         # k_0 confidence in mu_0
                "nu_0": np.full(N_cluster, 8),                          # nu_0 d.o.f. must be >d-1 
                "alpha": alpha                                          # starting alpha for dirichlet 
            },
            "alpha_0_": np.full(N_cluster, alpha/N_cluster),            # first values of alpha_k
            # this are the parts that will change
            "vec_z": vec_z,                                             # first random assignment
            "steps_done": 0,                                            # counter of steps done
            "suff_stats": {                                             # store the quantities to compute the posterior parameters
                "N_k": np.array([np.sum(vec_z==k) for k in range(N_cluster)]),      # count of points for cluster
                "bar_x_k": bar_x_k,                                                 # mean of data for cluster
                "S_k": S_k                                                          # the scatter matrix
            },
            "log_joint_p": np.nan                                       # log of joint probabilty 
        }

    def posterior_parameters(self):
        # compute the posterior parameters
        hyper = self.state["hyperparameters_"]
        suff = self.state["suff_stats"]
        k_n = hyper["k_0"] + suff["N_k"]
        nu_n = hyper["nu_0"] + suff["N_k"]
        mu_n = np.zeros(hyper["mu_0"].shape)
        for k in self.state["cluster_id_"]:
            mu_n[k] = (hyper["k_0"][k] * hyper["mu_0"][k] + suff["N_k"][k] * suff["bar_x_k"][k])/k_n[k]
        Psi_n = np.zeros(suff["S_k"].shape)
        for k in self.state["cluster_id_"]:
            Psi_n[k] = hyper["Psi_0"][k] + suff["S_k"][k] + hyper["k_0"][k]/k_n[k] * suff["N_k"][k] * np.outer((suff["bar_x_k"][k] - hyper["mu_0"][k]), (suff["bar_x_k"][k] - hyper["mu_0"][k]))
        
        return k_n, nu_n, mu_n, Psi_n

    def remove_suff_stats(self,index):
        # update the suff stats removing the point index
        kk = self.state["vec_z"][index]
        data_removed = np.delete(self.state["data_"], index, axis = 0)
        vec_z = np.delete(self.state["vec_z"], index)
        # update the suff_stats
        self.state["suff_stats"]["N_k"][kk] -= 1
        self.state["suff_stats"]["bar_x_k"][kk] = np.mean(data_removed[vec_z==kk], axis = 0)
        data_k = data_removed[vec_z==kk]
        diff = data_k - data_k.mean(axis=0)
        self.state["suff_stats"]["S_k"][kk] = diff.T @ diff
        
    def add_suff_stats(self, k_cluster):
        # update the suff_stats of cluster k
        data_k = self.state["data_"][self.state["vec_z"]==k_cluster]
        self.state["suff_stats"]["N_k"][k_cluster] += 1
        self.state["suff_stats"]["bar_x_k"][k_cluster] = np.mean(data_k, axis = 0)
        diff = data_k - data_k.mean(axis=0)
        self.state["suff_stats"]["S_k"][k_cluster] = diff.T @ diff

    def assign_new_zi(self, index):
        # calculate the new probability of assign point i to the clusters
        prob_zi = np.zeros(self.state["N_cluster_"])
        self.remove_suff_stats(index)           # update all the suff stat without point i
        for k in self.state["cluster_id_"]:
            first_term = (self.state["alpha_0_"][k] + self.state["suff_stats"]["N_k"][k])/(self.state["hyperparameters_"]["alpha"] + self.state["N_data_"] - 1)
            k_n, nu_n, mu_n, Psi_n = self.posterior_parameters()
            multivariate_t_n = multivariate_t(mu_n[k], (k_n[k]+1)/(k_n[k]*nu_n[k]) * Psi_n[k], df=nu_n[k]-2+1)
            second_term = multivariate_t_n.pdf(self.state["data_"][index])
            prob_zi[k] = first_term * second_term
        # normalize the probabilities
        prob_zi = prob_zi / np.sum(prob_zi)
        # assign the new z_i for the point
        new_k = rng.choice(self.state["N_cluster_"], p=prob_zi)
        # re add the point removed 
        self.state["vec_z"][index] = new_k
        self.add_suff_stats(new_k)
    
    def joint_prob(self):
        # compute the joint probability of the actual state of the sampler
        first_term = 0
        k_n, nu_n, mu_n, Psi_n = self.posterior_parameters()
        for i in range(self.state["N_data_"]):
            k = self.state["vec_z"][i]
            first_term += multivariate_t(mu_n[k], (k_n[k]+1)/(k_n[k]*nu_n[k]) * Psi_n[k], df=nu_n[k]-2+1).logpdf(self.state["data_"][i])
        second_term = 0
        for k in self.state["cluster_id_"]:
            second_term += math.lgamma(self.state["suff_stats"]["N_k"][k] + self.state["alpha_0_"][k])
        return first_term+second_term
    
    def make_a_step(self):
        # make a step of the sampler
        for ii in range(self.state["N_data_"]):
            self.assign_new_zi(ii)
        self.state["steps_done"] += 1
        self.state["log_joint_p"] = self.joint_prob()
    
    """
    def update_suff_stats(self):
        data = self.state["data_"]
        vec_z = self.state["vec_z"]
        # update the suff_stats to compute the posterior parameters
        self.state["suff_stats"]["N_k"] = np.array([
            np.sum(vec_z==k) for k in self.state["cluster_id_"]])
        
        self.state["suff_stats"]["bar_x_k"] = np.array([
            np.mean(data[vec_z==k]) for k in self.state["cluster_id_"]])
        
        self.state["suff_stats"]["S_k"] = np.array([
            np.cov(data[vec_z==k], rowvar=False, bias=True) * len(data[vec_z==k])
            for k in self.state["cluster_id_"]])
    """

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
    from matplotlib.lines import Line2D
    import os

    rng = np.random.default_rng(1313) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T, d_log_T, log_HR = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)

    xx = np.linspace(-4,7,256)              # linear space on log_T
    yy = np.linspace(-2.5, 3.5, 256)        # linear space on log_HR

    # define the bins
    bins_T = np.linspace(-4, 7, 51)
    width_T = np.diff(bins_T)[0]
    center_T = 0.5*(bins_T[1:] + bins_T[:-1])
    bottom_T = np.zeros(50, dtype=int)

    bins_HR = np.linspace(-2.5, 3.5, 51)
    width_HR = np.diff(bins_HR)[0]
    center_HR = 0.5*(bins_HR[1:] + bins_HR[:-1])
    bottom_HR = np.zeros(50, dtype=int)
    
    # try to initialise things
    # I have to define data so that data[ii] = [log_T[ii], log_HR[ii]] and see if it runs in 2D 
    data = np.array([log_T, log_HR]).T          #transpose to have format (N_point, N_dim = 2)
    
    fig0 = plt.figure("data: logT-logHR plane and marginals")
    ax  = fig0.add_subplot(111)
    axT = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    axH = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    # plot the main plane
    ax.scatter(data[:,0], data[:,1], marker = '.', color = 'C0', label = "Data")
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_xlim(-4,7)
    ax.set_ylabel("$\\log(HR)$")
    ax.set_ylim(-2.5,3.5)
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    # plot the marginal histograms
    count, _ = np.histogram(data[:,0], bins_T)
    axT.bar(center_T, count, width_T, color = 'C0')
    count, _ = np.histogram(data[:,1], bins_HR)
    axH.barh(center_HR, count, width_HR, color = 'C0')
    # set the right ticks and labels for histograms
    axT.set_ylabel("Counts")
    axH.set_xlabel("Counts")
    axT.tick_params(axis="x", labelbottom=False)
    axH.tick_params(axis="y", labelleft=False)
    axT.grid(linestyle = 'dashed')
    axT.set_axisbelow(True)
    axH.grid(linestyle = 'dashed')
    axH.set_axisbelow(True)
    plt.tight_layout()

    
    # initialize sampler
    
    # 3 clusters
    """    
    N_cluster = 3
    mu_0 = np.array([[-1,0],[1,0],[3.5,0]])
    cluster_color = ['g', 'purple', 'orange']
    conf_color = ['blue', 'black', 'red']
    """
    # 2 clusters    
    N_cluster = 2
    mu_0 = np.array([[-1,0],[3.5,0]])
    cluster_color = ['g', 'orange']
    conf_color = ['blue', 'red']
    

    sampler = state(N_cluster, data, mu_0=mu_0, alpha=1,rng=rng)
    _, _, mu_start, _ = sampler.posterior_parameters()

    fig1 = plt.figure("Initial assignment")
    ax = fig1.add_subplot(211)
    for k in sampler.state["cluster_id_"]:
        data_k = sampler.state["data_"][sampler.state["vec_z"] == k]
        ax.scatter(data_k[:,0], data_k[:,1], marker = '.', color = cluster_color[k], label = "Cluster number "+str(k))
    ax.set_title("Initial assignment")
    ax.set_ylabel("$\\log(HR)$")
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    # run n_step times
    run = False
    burn_in = 50
    n_samples = 5000     # steps of the sampler
    n_step = burn_in+n_samples
    mu_trace = np.zeros((n_samples+1, mu_0.shape[0], mu_0.shape[1]))   # save evolutions of E[mu]
    mu_trace[0] = mu_start                                          # save starting point of means 
    log_p = np.zeros(n_step)                                  # save evolution of joint probability of state

    if run == True:
        for i in range(burn_in):
            sampler.make_a_step()
            # save the mu and the log_p of the step to reconstruct the traces
            _, _, mu_n, _ = sampler.posterior_parameters()
            mu_trace[i+1] = mu_n
            log_p[i] = sampler.state["log_joint_p"]
            print(f"Step {sampler.state["steps_done"]} done")
        
        print("-------------------")
        print("Ended the burn-in, start to save samples")
        print("-------------------")
        # initialize items to be filled with samples
        mu_T = np.zeros((N_cluster, n_samples))          # first index = cluster, second index = iteration
        mu_HR = np.zeros((N_cluster, n_samples))
        sigma_T = np.zeros((N_cluster, n_samples))
        sigma_HR = np.zeros((N_cluster, n_samples))
        rho = np.zeros((N_cluster, n_samples))

        for i in range(n_samples):
            sampler.make_a_step()
            # save the mu and the log_p of the step to reconstruct the traces
            k_n, nu_n, mu_n, Psi_n = sampler.posterior_parameters()
            mu_trace[i+1] = mu_n
            log_p[burn_in+i] = sampler.state["log_joint_p"]
            # sample and save
            for k in sampler.state["cluster_id_"]:
                Sigma_j = invwishart(nu_n[k], Psi_n[k], seed=rng).rvs()                 # sample Sigma_j from the IW
                mu_j = multivariate_normal(mu_n[k], Sigma_j/k_n[k], seed=rng).rvs()     # sample mu_j from multi_normal
                mu_T[k,i] = mu_j[0]
                mu_HR[k,i] = mu_j[1]                                                    # write the two means
                sigma_T[k,i] = np.sqrt(Sigma_j[0,0])                                    # sigma on T
                sigma_HR[k,i] = np.sqrt(Sigma_j[1,1])                                   # sigma on HR
                rho[k,i] = Sigma_j[0,1]/(sigma_T[k,i]*sigma_HR[k,i])                    # correlation

            if sampler.state["steps_done"]/50 == np.floor(sampler.state["steps_done"]/50):
                print(f"Step {sampler.state["steps_done"]} done") # print only every 50 steps

            # maybe this can be a function of the sampler but try like this

        # after the running save the important parameters so we don't have to run it always
        # save vec_z assignment to cluster
        vec_z = sampler.state["vec_z"]
        np.savetxt(main_dir+"\\Gibbs\\vec_z_cluster_"+str(N_cluster)+"_step_"+str(n_step)+".txt", vec_z, header="vec_z (cluster "+str(N_cluster)+" step "+str(n_step)+")")
        # save joint probability evolution
        np.savetxt(main_dir+"\\Gibbs\\log_p_cluster_"+str(N_cluster)+"_step_"+str(n_step)+".txt", log_p, header="log_joint_p (cluster "+str(N_cluster)+" step "+str(n_step)+")")
        # save trace of means
        np.savez(main_dir+"\\Gibbs\\mu_trace_cluster_"+str(N_cluster)+"_step_"+str(n_step), mu_trace= mu_trace)
        # save posterior samples
        np.savez(main_dir+"\\Gibbs\\Posterior_samples_cluster_"+str(N_cluster)+"_step_"+str(n_step),
               mu_T=mu_T, mu_HR=mu_HR, sigma_T=sigma_T, sigma_HR=sigma_HR, rho=rho)

    # if we are not running the sampler search the saved files
    else:
        # load the vector of assignments
        vec_z = np.loadtxt(main_dir+"\\Gibbs\\vec_z_cluster_"+str(N_cluster)+"_step_"+str(n_step)+".txt")
        # load trace of joint prob 
        log_p = np.loadtxt(main_dir+"\\Gibbs\\log_p_cluster_"+str(N_cluster)+"_step_"+str(n_step)+".txt")
        # load file of traces of mean and extract vector 
        mu_trace_file = np.load(main_dir+"\\Gibbs\\mu_trace_cluster_"+str(N_cluster)+"_step_"+str(n_step)+".npz")
        mu_trace = mu_trace_file['mu_trace']
        # load posterior samples
        samples_file = np.load(main_dir+"\\Gibbs\\Posterior_samples_cluster_"+str(N_cluster)+"_step_"+str(n_step)+".npz")
        mu_T     = samples_file['mu_T']
        mu_HR    = samples_file['mu_HR']
        sigma_T  = samples_file['sigma_T']
        sigma_HR = samples_file['sigma_HR']
        rho      = samples_file['rho']
        print("Loaded the results of sampler (with "+str(N_cluster)+" clusters and "+str(n_step)+" step)")

    # look at the final assignments for points 
    ax1 = fig1.add_subplot(212, sharex=ax)
    for k in sampler.state["cluster_id_"]:
        data_k = sampler.state["data_"][vec_z == k]
        ax1.scatter(data_k[:,0], data_k[:,1], marker = '.', color = cluster_color[k], label = "Cluster number "+str(k))
    ax1.set_title("After "+str(n_step)+" step")
    ax1.set_xlabel("$\\log(T_{90})$")
    ax1.set_ylabel("$\\log(HR)$")
    ax1.grid(linestyle = 'dashed')
    ax1.set_axisbelow(True)
    plt.legend()
    plt.tight_layout()

    # plot the joint probability and check good burn-in
    fig2 = plt.figure("Joint probability trace")
    ax = fig2.add_subplot(111)
    ax.plot(log_p, '-', color='C0', label = "Joint probability")
    ax.set_xlabel("Steps")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("$\\log P$")
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    plt.close('all')

    fig2b = plt.figure("Joint probability trace zoom")
    ax = fig2b.add_subplot(111)
    ax.plot(log_p, marker='.', color='C0', label = "Joint probability")
    ax.axvline(burn_in, color ='r', linestyle='dashed', label='Burn-in')
    ax.set_xlabel("Steps")
    ax.set_xlim(-5, 2*burn_in)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("$\\log P$")
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    plt.show()
    exit()

    """
    # I like this but maybe not really useful 
    fig5 = plt.figure("Mean trace")
    ax = fig5.add_subplot(111)
    for k in sampler.state["cluster_id_"]:
        mu_k = mu_trace[:,k,:]
        ax.plot(mu_k[:,0], mu_k[:,1], '-o', color = cluster_color[k])
        ax.plot(mu_k[0,0], mu_k[0,1], 'o', color = cluster_color[k], mec='b')
        ax.plot(mu_k[-1,0], mu_k[-1,1], 'o', color = cluster_color[k], mec='r')
    legend_elements = [
        Line2D([0], [0], marker = 'o', color=cluster_color[0], label = "Mean cluster 0"),
        Line2D([0], [0], marker = 'o', color=cluster_color[1], label = "Mean cluster 1"),
        Line2D([0], [0], linewidth=0, marker = 'o', mfc='w', mec = 'b', label = "Starting means"),
        Line2D([0], [0], linewidth=0, marker = 'o', mfc='w', mec = 'r', label = "Ending means"),] 
    ax.set_title("Means traces")
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("$\\log(HR)$")
    plt.legend(handles=legend_elements)
    """

    par_name = ["$\\mu_T$", "$\\sigma_T$", "$\\mu_{HR}$", "$\\sigma_{HR}$", "$\\rho$"]
    thinning = 20
    # autocorrelation for each set of parameters
    fig3 = plt.figure("Parameters autocorrelation ("+str(N_cluster)+" clusters)", figsize = (6,6))
    for k in range(N_cluster):
        parameters = [mu_T[k], sigma_T[k], mu_HR[k], sigma_HR[k], rho[k]]
        for i in range(len(parameters)):
            ax = fig3.add_subplot(5, N_cluster, k+N_cluster*i+1)
            if k+N_cluster*i == np.arange(N_cluster)[k]:
                ax.set_title("Cluster "+str(k))
            ax.plot(autocorrelation(parameters[i]), '.', color = 'C0', label = 'Autocorrelation')
            ax.set_ylabel(par_name[i])
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    fig3b = plt.figure("Parameters autocorrelation ("+str(N_cluster)+" clusters) zoom", figsize = (6,6))
    for k in range(N_cluster):
        parameters = [mu_T[k], sigma_T[k], mu_HR[k], sigma_HR[k], rho[k]]
        for i in range(len(parameters)):
            ax = fig3b.add_subplot(5, N_cluster, k+N_cluster*i+1)
            if k+N_cluster*i == np.arange(N_cluster)[k]:
                ax.set_title("Cluster "+str(k))
            ax.plot(autocorrelation(parameters[i]), '.', color = 'C0', label = 'Autocorrelation')
            ax.axvline(thinning, color='r', linestyle = 'dashed', label = 'Thinning')
            ax.set_ylabel(par_name[i])
            ax.set_xlim(-10, 2*thinning)
    ax.set_xlabel("Iteration")
    plt.tight_layout()

    # thinning the samples    
    par_mu_T = mu_T[:,::thinning]
    par_sigma_T = sigma_T[:,::thinning]
    par_mu_HR = mu_HR[:,::thinning]
    par_sigma_HR = sigma_HR[:,::thinning]
    par_rho = rho[:,::thinning]
    
    print(f"After burn-in and thinning we have {len(par_mu_T[0,:])} samples")
    
    # find peaks and credible interval
    par_val = np.zeros((N_cluster,len(par_name)))
    d_par_plus = np.zeros((N_cluster,len(par_name)))
    d_par_minus = np.zeros((N_cluster,len(par_name)))
    for k in range(N_cluster):
        parameters = [par_mu_T[k], par_sigma_T[k], par_mu_HR[k], par_sigma_HR[k], par_rho[k]]
        for i in range(len(par_name)):
            counts_i, bins_i = np.histogram(parameters[i], bins = 30, density=True)
            center_i = 0.5*(bins_i[1:] + bins_i[:-1])
            par_val[k,i], d_par_plus[k,i], d_par_minus[k,i] = errors_around_peak(center_i, counts_i)

    # histogram of the parameters
    for k in range(N_cluster):
        parameters = [par_mu_T[k], par_sigma_T[k], par_mu_HR[k], par_sigma_HR[k], par_rho[k]]
        fig4 = plt.figure("Parameters histogram (cluster "+str(k)+")", figsize = (6,6))
        for i in range(len(par_name)):
            ax = fig4.add_subplot(3, 2, i+1)
            counts_i, bins_i = np.histogram(parameters[i], bins = 30, density=True)
            ax.stairs(counts_i, bins_i, color = 'C0', linewidth = 1.5, baseline=0)
            ax.axvline(par_val[k,i], color = 'green', linestyle='dashed')
            ax.axvline(par_val[k,i]+d_par_plus[k,i], color = 'orange', linestyle='dashed')
            ax.axvline(par_val[k,i]-d_par_minus[k,i], color = 'orange', linestyle='dashed')
            ax.set_xlabel(par_name[i])
            delta = 2*np.diff(bins_i)[0]
            ax.set_xlim(np.min(bins_i)-delta, np.max(bins_i)+delta)
            ax.set_ylim(0, np.max(counts_i) * 1.1)

        ax_leg = fig4.add_subplot(3,2,6)
        legend_elements = [
            Line2D([0], [0], color='C0', linewidth=1.5, label='Marginalized posterior samples'),
            Line2D([0], [0], color='green', linestyle='dashed', label='Median value'),
            Line2D([0], [0], color='orange', linestyle='dashed', label='90% confidence interval'),]
        ax_leg.legend(handles=legend_elements, loc='center')
        ax_leg.axis('off')  # Hide axes, ticks, spines and background
        plt.tight_layout()
        
        # rename the fig to save them later
        if k==0:
            fig_hist_par_0 = fig4
        if k==1:
            fig_hist_par_1 = fig4
        if k==2:
            fig_hist_par_2 = fig4

    # confidence levels in the plane
    from matplotlib.patches import Ellipse
    scale = 2       # how many sigmas we want to represent? scale = 2 has ~68% of data     

    # function to draw confidence leves around ellipses
    def draw_band(ax, center, width_out, height_out, width_in, height_in, angle, color, alpha=0.4, zorder=2):
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
        xx, yy = np.meshgrid(np.linspace(center[0]-width_out, center[0]+width_out, 1600),
                         np.linspace(center[1]-height_out, center[1]+height_out, 1600))
    
        dx, dy = xx - center[0], yy - center[1]
        x_rot =  cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy
    
        inside_out = (x_rot/(width_out/2))**2  + (y_rot/(height_out/2))**2 <= 1
        inside_in  = (x_rot/(width_in/2))**2  + (y_rot/(height_in/2))**2  <= 1
    
        band = (inside_out & ~inside_in).astype(float)
        band[band == 0] = np.nan  # transparent outside
    
        cf = ax.contourf(xx, yy, band, levels=[0.5, 1.5], colors=[color], alpha=alpha, zorder=zorder)
        cf.set_clip_path(ax.patch)
       
    bottom_T = np.zeros(center_T.shape)
    bottom_HR = np.zeros(center_HR.shape)

    # the plot
    fig5b = plt.figure("Clusters in plane log_T log_HR")
    ax = fig5b.add_subplot(111)
    axT = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    axH = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    for k in range(N_cluster):
        data_k = sampler.state["data_"][vec_z == k]
        ax.scatter(data_k[:,0], data_k[:,1], marker = '.', color = cluster_color[k], zorder = 1, alpha=0.25, label="Cluster "+str(k))
        count_T, _ = np.histogram(data_k[:,0], bins_T)
        axT.bar(center_T, count_T, width_T, color = cluster_color[k], bottom=bottom_T, alpha=0.5)
        bottom_T += count_T
        count_HR, _ = np.histogram(data_k[:,1], bins_HR)
        axH.barh(center_HR, count_HR, width_HR, color = cluster_color[k], left=bottom_HR, alpha=0.5)
        bottom_HR += count_HR
        
        # find center of the cluster
        center = np.array([par_mu_T[k,:], par_mu_HR[k,:]]).T
        c_m = np.percentile(center, 50, axis=0)
        ax.scatter(c_m[0], c_m[1], marker = '.', color = conf_color[k], zorder = 2, alpha=1)
        """
        axT.plot(c_m[0], 150, marker = 'v', color = conf_color[k], zorder = 2, alpha=1)
        axH.plot(250, c_m[1], marker = '<', color = conf_color[k], zorder = 2, alpha=1)
        """
        axT.axvline(c_m[0], color = conf_color[k], zorder = 2, alpha=1)
        axH.axhline(c_m[1], color = conf_color[k], zorder = 2, alpha=1)
        
        eig = np.zeros(center.shape)
        angle = np.zeros(center.shape[0])
        for i in range(len(par_mu_T[0,:])):
            Sigma = np.array([[par_sigma_T[k,i]**2, par_rho[k,i]*par_sigma_T[k,i]*par_sigma_HR[k,i]],
                              [par_rho[k,i]*par_sigma_T[k,i]*par_sigma_HR[k,i], par_sigma_HR[k,i]**2]])
            eig[i], eigvec = np.linalg.eigh(Sigma)
            angle[i] = np.arctan2(eigvec[1,1], eigvec[0,1]) * 180/np.pi    # rotation angle
        
        # find median values
        eig_p = np.percentile(eig, [5,50,95], axis=0)
        angle_m = np.degrees(np.arctan2(np.mean(np.sin(np.radians(angle))), np.mean(np.cos(np.radians(angle))))) 
        width = scale*2*np.sqrt(eig_p[:,-1])                               # major axis
        height = scale*2*np.sqrt(eig_p[:,0])                               # minor axis
        # 5% ellipse
        ellipse = Ellipse(xy=c_m, width=width[0], height=height[0], angle=angle_m, fill=False, color = conf_color[k], linestyle='dashed')
        ax.add_patch(ellipse)
        # 95% ellipse
        ellipse = Ellipse(xy=c_m,  width=width[2], height=height[2], angle=angle_m, fill=False, color = conf_color[k], linestyle='dashed')
        ax.add_patch(ellipse)
        # draw the confidence band
        draw_band(ax, c_m, width[2], height[2], width[0], height[0], angle_m, conf_color[k], alpha=0.5, zorder=2)

        # median ellipse
        ellipse = Ellipse(xy=c_m, width=width[1], height=height[1], angle=angle_m, fill=False, color=conf_color[k], alpha=1, zorder=3)
        ax.add_patch(ellipse)

        # confidence band on the marginals, nope, too much
        """
        axT.axvline(c_m[0]-width[1]/2, color = conf_color[k], zorder = 2, alpha=1, linestyle='dashed') 
        axT.axvline(c_m[0]+width[1]/2, color = conf_color[k], zorder = 2, alpha=1, linestyle='dashed') 
        axH.axhline(c_m[1]-height[1]/2, color = conf_color[k], zorder = 2, alpha=1, linestyle='dashed') 
        axH.axhline(c_m[1]+height[1]/2, color = conf_color[k], zorder = 2, alpha=1, linestyle='dashed') 
        """
    # main plane labels
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_xlim(-4,7)
    ax.set_ylabel("$\\log(HR)$")
    ax.set_ylim(-2.5,3.5)
    ax.legend(bbox_to_anchor=(1.1, 1.05, 0.25, 0.25))
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    # set the right ticks and labels for histograms
    axT.set_ylabel("Counts")
    axH.set_xlabel("Counts")
    axT.tick_params(axis="x", labelbottom=False)
    axH.tick_params(axis="y", labelleft=False)
    axT.grid(linestyle = 'dashed')
    axT.set_axisbelow(True)
    axH.grid(linestyle = 'dashed')
    axH.set_axisbelow(True)
    plt.tight_layout()

    
    # end of the point: show, save or close all the open figures
    """
    plt.show()
    exit()
    """
    path = main_dir+"\\Results\\5\\"+str(N_cluster)+"_step_"+str(n_step)
    # Check if the results dir exists
    if not os.path.exists(path):
        os.mkdir(path)
    
    fig0.savefig(main_dir+"\\Results\\Data_T_HR_plane.png", dpi = 600)
    fig1.savefig(path+"\\Assignments_start_end.png", dpi = 600)
    fig2.savefig(path+"\\Joint_probability.png", dpi = 600)
    fig2b.savefig(path+"\\Joint_probability_zoom.png", dpi = 600)
    fig3.savefig(path+"\\Parameters_autocorr.png", dpi = 600)
    fig3b.savefig(path+"\\Parameters_autocorr_zoom.png", dpi = 600)
    fig_hist_par_0.savefig(path+"\\Parameters_hist_0.png", dpi = 600)
    fig_hist_par_1.savefig(path+"\\Parameters_hist_1.png", dpi = 600)
    if N_cluster > 2:
        fig_hist_par_2.savefig(path+"\\Parameters_hist_2.png", dpi = 600)

    fig5b.savefig(path+"\\Plane_confidence.png", dpi = 600)

    header = "par_val d_par_plus d_par_minus for [mu_T, sigma_T, mu_HR, sigma_HR, rho]"
    np.savetxt(path+"\\parameters_values_0.txt", np.array([par_val[0], d_par_plus[0], d_par_minus[0]]).T, header=header)
    np.savetxt(path+"\\parameters_values_1.txt", np.array([par_val[1], d_par_plus[1], d_par_minus[1]]).T, header=header)
    if N_cluster > 2:
        np.savetxt(path+"\\parameters_values_2.txt", np.array([par_val[2], d_par_plus[2], d_par_minus[2]]).T, header=header)