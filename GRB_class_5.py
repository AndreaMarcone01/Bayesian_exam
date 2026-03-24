# Python script for point 1a: find parameters of the classification model

import numpy as np
from scipy.special import xlogy
from pdf_analysis import errors_around_peak
from scipy.stats import multivariate_normal, multivariate_t

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

def gauss_2D(x, theta):
    """Pdf of a 2D gaussian.
    
    Args:
        x (array): multidimensional variable
        theta (array): parameters in order [mu_x, mu_y, sigma_x, sigma_y, rho]
        
    Returns:
        pdf (array): probability density function, normalized to 1
    """

    sigma_x = theta[2]
    sigma_y = theta[3]
    rho = theta[4]
    cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                    [rho*sigma_x*sigma_y, sigma_y**2]])
    
    multi = multivariate_normal(mean=np.array([theta[0], theta[1]]), cov=cov)
    pdf = multi.pdf(x)
    return pdf

def weighted_2D_normal(x, theta):
    """Weighted sum of two 2D normal.
    
    Args:
        x (array): independent variable
        theta (array): parameter array, in order [w, theta_1, theta_2]
        
    Returns:
        total model (array): probability density function, normalized to 1
    """

    w = theta[0]
    theta_1 = theta[1:6]
    theta_2 = theta[6:11]

    normal_1 = gauss_2D(x, theta_1)
    normal_2 = gauss_2D(x, theta_2)
    model = w * normal_1 + (1-w) * normal_2
    return model

class state():
    def __init__(self, N_cluster, data, mu_0, alpha, rng):
        vec_z = rng.choice(N_cluster, size=data.shape[0])
        identity = np.diag(np.full(data.shape[1],1))
                
        # compute some suff_stats that can't be done in a line
        bar_x_k = np.zeros((N_cluster, data.shape[1]))      # firts index 0=log_T, 1=log_HR, second index = k
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
            "vec_z": vec_z,                                             # first random assignemnt
            "steps_done": 0,                                            # counter of steps done
            "suff_stats": {                                             # store the quantities to compute the posterior parameters
                "N_k": np.array([np.sum(vec_z==k) for k in range(N_cluster)]),              # first count of points for cluster
                "bar_x_k": bar_x_k,                                    # mean of data for cluster
                "S_k": S_k                            # compute the scatter matrix
            }
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
        self.add_suff_stats(index, new_k)

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
        
    def add_suff_stats(self, ii, kk):
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
    
    def make_a_step(self):
        # make a step of the sampler
        for ii in range(self.state["N_data_"]):
            self.assign_new_zi(ii)
        self.state["steps_done"] += 1
    """
    # don't know if this are used
    def count_in_cluster(self, exclude_index):
        # count the data for cluster without point i
        vec_z_minus_i = np.delete(self.state["vec_z"], exclude_index)   # remove point i from vec_z
        N_k_minus_i = np.zeros(self.state["N_cluster_"])                # initialise counts once removed x_i
        for k in range(len(N_k_minus_i)):
            N_k_minus_i[k] = np.sum(vec_z_minus_i==k)                   # counts once removed x_i
        return N_k_minus_i

    def update_suff_stats(self, index=-1):
        if index > -0.1:
            data = np.delete(self.state["data_"], index)
            vec_z = np.delete(self.state["vec_z"], index)
        else:
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
        
    def update_alphas(self, exclude_index):
        # find new alphas (don't know if it's used)
        for k in self.state["cluster_id_"]:
            self.state["alpha_k"][k] = self.state["alpha_"]/self.state["N_cluster_"] + self.count_in_cluster(exclude_index)[k]
    """

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(1313) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T, d_log_T, log_HR = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)

    counts, xedge, yedge = np.histogram2d(log_T, log_HR, bins = 25)
    hist, _, _ = np.histogram2d(log_T, log_HR, bins = 25, density = True)
    counts = counts.T
    xcenter = np.asarray(0.5*(xedge[1:] + xedge[:-1]))
    ycenter = np.asarray(0.5*(yedge[1:] + yedge[:-1]))
    dA = np.diff(xcenter)[0] * np.diff(ycenter)[0]
    xc, yc = np.meshgrid(xcenter, ycenter)  # C = ax.contourf(xc, yc, counts) to plot
                                            # plt.colorbar(C)
    centers = np.dstack((xc, yc))           # points on which we evaluate the model

    xx = np.linspace(-4,7,256)
    yy = np.linspace(-2.5, 3.5, 256)
    x, y = np.meshgrid(xx, yy)              # to plot smooth
    pos_plot = np.dstack((x, y))            # what needs to be passed to model function

    par_name = ["w", "mu_1_T", "mu_1_HR", "sigma_1_T", "sigma_1_HR", "rho1", 
              "mu_2_T", "mu_2_HR", "sigma_2_T", "sigma_2_HR", "rho2"]
    bounds = np.array([[0,1], 
                       [-4,7], [-2.5, 3.5], [0.01, 3], [0.01, 3], [0,1],
                       [-4,7], [-2.5, 3.5], [0.01, 3], [0.01, 3], [-0.99,0.99]])

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

    # thinked it would do better but no problem. Let's see the parameters
    k_n, nu_n, mu_n, Psi_n = initial_state.posterior_parameters()
    cov = [Psi_n[i]/(nu_n[i] - 1 - 1) for i in range(len(k_n))]
    print(f"The means are {mu_n}")                  # The means are [-0.0444695   3.60561329]
    print(f"And the covariancies {cov}")            # And the covariancies [2.1759942246549575, 0.8944213457721013]
    plt.show()
    """

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
    cluster_color = ['g', 'orange']

    fig4 = plt.figure("Initial assignment")
    ax = fig4.add_subplot(111)
    for k in sampler.state["cluster_id_"]:
        data_k = sampler.state["data_"][sampler.state["vec_z"] == k]
        ax.scatter(data_k[:,0], data_k[:,1], marker = '.', color = cluster_color[k], label = "Cluster number "+str(k))
    ax.set_title("Initial assignment")
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("$\\log(HR)$")
    plt.legend()

    # run n_step times
    n_step = 2
    for _ in range(n_step):
        sampler.make_a_step()
        print(f"Step {sampler.state["steps_done"]} done")


    k_n, nu_n, mu_n, Psi_n = sampler.posterior_parameters()
    cov = [Psi_n[i]/(nu_n[i] - 1 - 1) for i in range(len(k_n))]
    print(f"The means are {mu_n}")                  # The means are [-0.0444695   3.60561329]
    print(f"And the covariancies {cov}")