# Python script for point 1a: find parameters of the classification model

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
        vec_z = rng.choice(N_cluster, size=data.shape[0])
        identity = np.diag(np.full(data.shape[1],1))
                
        # compute some suff_stats that can't be done in a line
        bar_x_k = np.zeros((N_cluster, data.shape[1]))      # first index 0=log_T, 1=log_HR, second index = k
        for k in range(N_cluster):
            bar_x_k[k] = np.mean(data[vec_z == k], axis = 0)

        if bar_x_k.shape != mu_0.shape:
            print("Something is wrong, check the initial mu_0!")
            print("Exiting...")
            exit()

        S_k = np.zeros((N_cluster, data.shape[1], data.shape[1]))
        for k in range(N_cluster):
            data_k = data[vec_z==k]
            diff = data_k - data_k.mean(axis=0)
            S_k[k] =  diff.T @ diff 

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
            "alpha_0": np.full(N_cluster, alpha/N_cluster),             # first values of alpha_k
            "vec_z": vec_z,                                             # first random assignment
            "steps_done": 0,                                            # counter of steps done
            "suff_stats": {                                             # store the quantities to compute the posterior parameters
                "N_k": np.array([np.sum(vec_z==k) for k in range(N_cluster)]),              # first count of points for cluster
                "bar_x_k": bar_x_k,                                    # mean of data for cluster
                "S_k": S_k                            # compute the scatter matrix
            },
            "log_joint_p": np.nan
        }

    def assign_new_zi(self, index):
        # calculate the new probability of assign point i to the clusters
        prob_zi = np.zeros(self.state["N_cluster_"])
        self.remove_suff_stats(index)           # update all the suff stat without point i
        for k in self.state["cluster_id_"]:
            first_term = (self.state["alpha_0"][k] + self.state["suff_stats"]["N_k"][k])/(self.state["hyperparameters_"]["alpha"] + self.state["N_data_"] - 1)
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

    def remove_suff_stats(self,index):
        kk = self.state["vec_z"][index]
        data_removed = np.delete(self.state["data_"], index, axis = 0)
        vec_z = np.delete(self.state["vec_z"], index)
        # update the suff_stats
        self.state["suff_stats"]["N_k"][kk] -= 1
        self.state["suff_stats"]["bar_x_k"][kk] = np.mean(data_removed[vec_z==kk], axis = 0)
        data_k = data_removed[vec_z==kk]
        diff = data_k - data_k.mean(axis=0)
        self.state["suff_stats"]["S_k"][kk] = diff.T @ diff
        
    def add_suff_stats(self, kk):
        # update the suff_stats of cluster kk
        data = self.state["data_"]
        vec_z = self.state["vec_z"]
        self.state["suff_stats"]["N_k"][kk] += 1
        self.state["suff_stats"]["bar_x_k"][kk] = np.mean(data[vec_z==kk], axis = 0)
        data_k = data[vec_z==kk]
        diff = data_k - data_k.mean(axis=0)
        self.state["suff_stats"]["S_k"][kk] = diff.T @ diff

    def posterior_parameters(self):
        # compute the posterior parameters
        hyper = self.state["hyperparameters_"]
        suff = self.state["suff_stats"]
        k_n = hyper["k_0"] + suff["N_k"]
        nu_n = hyper["nu_0"] + suff["N_k"]
        mu_n = (hyper["k_0"] * hyper["mu_0"] + suff["N_k"] * suff["bar_x_k"])/k_n
        Psi_n = np.zeros(suff["S_k"].shape)
        for k in self.state["cluster_id_"]:
            Psi_n[k] = hyper["Psi_0"][k] + suff["S_k"][k] + hyper["k_0"][k]/k_n[k] * suff["N_k"][k] * np.outer((suff["bar_x_k"][k] - hyper["mu_0"][k]), (suff["bar_x_k"][k] - hyper["mu_0"][k]))
        
        return k_n, nu_n, mu_n, Psi_n
    
    def joint_prob(self):
        # compute the joint probability of the actual state of the sampler
        first_term = 0
        k_n, nu_n, mu_n, Psi_n = self.posterior_parameters()
        for i in range(self.state["N_data_"]):
            k = self.state["vec_z"][i]
            first_term += multivariate_t(mu_n[k], (k_n[k]+1)/(k_n[k]*nu_n[k]) * Psi_n[k], df=nu_n[k]-2+1).logpdf(self.state["data_"][i])
        second_term = 0
        for k in self.state["cluster_id_"]:
            second_term += math.lgamma(self.state["suff_stats"]["N_k"][k] + self.state["alpha_0"][k])
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import os

    rng = np.random.default_rng(1313) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T, d_log_T, log_HR = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)

    xx = np.linspace(-4,7,256)              # linear space on log_T
    yy = np.linspace(-2.5, 3.5, 256)        # linear space on log_HR
    
    
    # try to initialise things
    # I have to define data so that data[ii] = [log_T[ii], log_HR[ii]] and see if it runs in 2D 
    data = np.array([log_T, log_HR]).T          #transpose to have format (N_point, N_dim = 2)
    
    """
    # 3 clusters    
    N_cluster = 3
    mu_0 = np.array([[-1,0],[1,0],[3.5,0]])
    """

    # 2 clusters    
    N_cluster = 2
    mu_0 = np.array([[-1,0],[3.5,0]])

    sampler = state(N_cluster, data, mu_0=mu_0, alpha=1,rng=rng)
    _, _, mu_start, _ = sampler.posterior_parameters()
    cluster_color = ['g', 'orange']

    fig4 = plt.figure("Initial assignment")
    ax = fig4.add_subplot(211)
    for k in sampler.state["cluster_id_"]:
        data_k = sampler.state["data_"][sampler.state["vec_z"] == k]
        ax.scatter(data_k[:,0], data_k[:,1], marker = '.', color = cluster_color[k], label = "Cluster number "+str(k))
    ax.set_title("Initial assignment")
    ax.set_ylabel("$\\log(HR)$")
    plt.legend()

    # run n_step times
    run = True
    burn_in = 6
    n_samples = 15     # steps of the sampler
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
            print(f"Step {sampler.state["steps_done"]} done")
            # sample and save
            for k in range(N_cluster):
                Sigma_j = invwishart(nu_n[k], Psi_n[k], seed=rng).rvs()                 # sample Sigma_j from the IW
                mu_j = multivariate_normal(mu_n[k], Sigma_j/k_n[k], seed=rng).rvs()     # sample mu_j from multi_normal
                mu_T[k,i] = mu_j[0]
                mu_HR[k,i] = mu_j[1]                                                    # write the two means
                sigma_T[k,i] = np.sqrt(Sigma_j[0,0])                                    # sigma on T
                sigma_HR[k,i] = np.sqrt(Sigma_j[1,1])                                   # sigma on HR
                rho[k,i] = Sigma_j[0,1]/(sigma_T[k,i]*sigma_HR[k,i])                    # correlation
    
            print("Saved samples")
            print("-------------------")

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


    ax1 = fig4.add_subplot(212, sharex=ax)
    for k in sampler.state["cluster_id_"]:
        data_k = sampler.state["data_"][vec_z == k]
        ax1.scatter(data_k[:,0], data_k[:,1], marker = '.', color = cluster_color[k], label = "Cluster number "+str(k))
    ax1.set_title("After "+str(n_step)+" step")
    ax1.set_xlabel("$\\log(T_{90})$")
    ax1.set_ylabel("$\\log(HR)$")
    plt.legend()
    plt.tight_layout()

    fig5 = plt.figure("Joint probability trace")
    ax = fig5.add_subplot(111)
    ax.plot(log_p, '-o', color='C0', label = "Joint probability")
    ax.axvline(burn_in, color ='r', linestyle='dashed', label='Burn-in')
    ax.set_xlabel("Steps")
    ax.set_ylabel("$\\log P$")
    plt.legend()
    plt.show()
    exit()

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
    
    #plt.show()

    # some code here could be useful for 2D plot
    """
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    C   = ax.contourf(x,y, pdf, 100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(C)
    plt.show()
    
    fig0 = plt.figure("logT-logHR plane")
    plt.scatter(log_T, log_HR, marker = '.', label = 'Data')
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("$\\log(HR)$")
    plt.tight_layout()
    
    fig1 = plt.figure("logT-logHR plane and marginals")
    ax = fig1.add_subplot(2,2,3)
    axT = fig1.add_subplot(2,2,1, sharex = ax)
    axH = fig1.add_subplot(2,2,4, sharey = ax)
    axT.hist(log_T, bins=50)
    axH.hist(log_HR, bins=50, orientation="horizontal")
    ax.scatter(log_T, log_HR, marker = '.', label = 'Data')
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("$\\log(HR)$")
    plt.tight_layout()
    
    fig2 = plt.figure("HR distribution with short/long")
    plt.hist(log_HR[log_T<1.67], bins = 30, color='g', label='Short GRB', fill = False, histtype='step')
    plt.hist(log_HR[log_T>1.67], bins = 30, color='orange', label='Long GRB', fill = False, histtype='step')
    plt.xlabel("$\\log(HR)$")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    """

    # some code here could be important to visualize the marginalize classification
    """
    # try to use this to verify algorithm on log_T that I already know the results
    initial_state = state(2, log_T, 1, rng)
    cluster_color = ['g', 'orange']
    bins = np.linspace(-4, 7, 51)
    width = np.diff(bins)[0]
    center = 0.5*(bins[1:] + bins[:-1])

    fig3 = plt.figure("Test on log_T, initial assignment")
    ax = fig3.add_subplot(211)
    bottom = np.zeros(50, dtype=int)
    for k in initial_state.state["cluster_id_"]:
        data_k = initial_state.state["data_"][initial_state.state["vec_z"] == k]
        count, _ = np.histogram(data_k, bins)
        ax.bar(center, count, width, color = cluster_color[k], label = "Cluster number "+str(k), bottom=bottom)
        bottom += count
    ax.set_title("Initial assignment")
    ax.set_ylabel("Counts")
    plt.legend()
    
    # run 5 times
    n_step = 25
    for _ in range(n_step):
        initial_state.make_a_step()
        print(f"Step {initial_state.state["steps_done"]} done")

    ax1 = fig3.add_subplot(212, sharex = ax)
    bottom = np.zeros(50, dtype=int)
    for k in initial_state.state["cluster_id_"]:
        data_k = initial_state.state["data_"][initial_state.state["vec_z"] == k]
        count, _ = np.histogram(data_k, bins)
        ax1.bar(center, count, width, color = cluster_color[k], label = "Cluster number "+str(k), bottom=bottom)
        bottom += count
    ax1.set_xlabel("$\\log(T_{90})$")
    ax1.set_title("After "+str(n_step)+" steps")
    ax1.set_ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    """