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
    def __init__(self, N_cluster, data, alpha, rng):
        vec_z = rng.choice(N_cluster, size=len(data))
        identity = np.diag(np.full(1,1))
        self.state = {
            "N_cluster_": N_cluster,                                    # number of cluster (=3)
            "cluster_id_": range(N_cluster),                            # id of cluster [0,1,2]
            "data_" : data,                                             # data to assign
            "N_data_": len(data),                                       # number of points
            "hyperparameters_": {
                "mu_0": np.array([-1,1,3.5]),               # mu_0 for every cluster [[-1,0],[1,0],[3.5,0]]
                "Psi_0": np.array([[identity],[identity],[identity]]),  # covariance_0 for every cluster
                "k_0": np.full(N_cluster, 0.01),                    # k_0 confidence in mu_0
                "nu_0": np.full(N_cluster, 3),                          # nu_0 d.o.f. must be >d-1 
                "alpha": alpha                                          # starting alpha for dirichlet 
            },
            "alpha_0": np.full(N_cluster, alpha/N_cluster),             # first values of alpha_k
            "vec_z": vec_z,                                             # first random assignemnt
            "steps_done": 0,                                            # counter of steps done
            "suff_stats": {                                             # store the quantities to compute the posterior parameters
                "N_k": np.array([np.sum(vec_z==k) for k in range(N_cluster)]),              # first count of points for cluster
                "bar_x_k": np.array([np.mean(data[vec_z==k]) for k in range(N_cluster)]),   # mean of data for cluster
                "S_k": np.array([np.cov(data[vec_z==k], rowvar=False, bias=True) * len(data[vec_z==k])
                                   for k in range(N_cluster)]),                             # compute the scatter matrix
            }
        }

    def assign_new_zi(self, index):
        # calculate the new probability of assign point i to the clusters
        prob_zi = np.zeros(self.state["N_cluster_"])
        self.remove_suff_stats(index)           # update all the suff stat without point i
        for k in self.state["cluster_id_"]:
            first_term = (self.state["alpha_0"][k] + self.state["suff_state"]["N_k"])/(self.state["hyperparameters_"]["alpha"] + self.state["N_data_"] - 1)
            k_n, nu_n, mu_n, Psi_n = self.posterior_parameters()
            multivariate_t_n = multivariate_t(mu_n[k], (k_n[k]+1)/(k_n[k]*nu_n[k]) * Psi_n[k], df=nu_n[k]-2+1)
            second_term = multivariate_t_n.pdf(self.state["data_"][index])
            prob_zi[k] = first_term * second_term
        # normalize the probabilities
        prob_zi = prob_zi / np.sum(prob_zi)
        print(prob_zi)
        # assign the new z_i for the point
        new_k = rng.choice(self.state["N_cluster_"], p=prob_zi)
        # re add the point removed 
        self.state["vec_z"][index] = new_k
        self.add_suff_stats(index, new_k)

    def remove_suff_stats(self,index):
        kk = self.state["vec_z"][index]
        data = np.delete(self.state["data_"], index)
        vec_z = np.delete(self.state["vec_z"], index)
        # update the suff_stats
        self.state["suff_stats"]["N_k"][kk] -= 1
        self.state["suff_stats"]["bar_x_k"][kk] = np.mean(data[vec_z==kk])
        self.state["suff_stats"]["S_k"][kk] = np.cov(data[vec_z==kk], rowvar=False, bias=True) * len(data[vec_z==kk])
        
    def add_suff_stats(self, ii, kk):
        # update the suff_stats of cluster kk
        data = self.state["data_"]
        vec_z = self.state["vec_z"]
        self.state["suff_stats"]["N_k"][kk] += 1
        self.state["suff_stats"]["bar_x_k"][kk] = np.mean(data[vec_z==kk])
        self.state["suff_stats"]["S_k"][kk] = np.cov(data[vec_z==kk], rowvar=False, bias=True) * len(data[vec_z==kk])

    def posterior_parameters(self):
        # compute the posterior parameters
        hyper = self.state["hyperparameters_"]
        suff = self.state["suff_stats"]
        k_n = hyper["k_0"] + suff["N_k"]
        nu_n = hyper["nu_0"] + suff["N_k"]
        mu_n = (hyper["k_0"] * hyper["mu_0"] + suff["N_k"] * suff["bar_x_k"])/k_n
        Psi_n = np.array([
            (hyper["Psi_0"][k] + suff["S_k"][k] + hyper["k_0"][k]/k_n[k] * suff["N_k"][k] * 
             np.outer((suff["bar_x_k"][k] - hyper["mu_0"][k]), (suff["bar_x_k"][k] - hyper["mu_0"][k])))
            for k in self.state["cluster_id_"]])
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

    # build theta_0
    theta_1 = np.array([-0.5, 0, 1, 1, 0.5])
    theta_2 = np.array([3.5, 0, 1, 1, 0.5])
    theta = np.array([0.5])
    theta_0 = np.append(np.append(theta, theta_1), theta_2)

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

    # try to initialise things

    initial_state = state(3, log_T, 1, rng)
    # try to use this to verify algorithm on log_T that I already know the results
    # then i have to define data so that data[ii] = [log_T[ii], log_HR[ii]]
    # and see if it runs in 2D 