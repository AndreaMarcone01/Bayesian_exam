import numpy as np

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(111) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))
    log_T90, d_log_T90, Hardness_R = np.loadtxt(main_dir+"\\Data\\GRB_data.txt", unpack=True)

    nbins = 50
    bins = np.linspace(-4, 7, nbins)
    xx = np.linspace(-4,7,100)
    hist, edges = np.histogram(log_T90, bins = bins, density = True)    # used in plot
    counts, _ = np.histogram(log_T90, bins = bins)                      # used for the real analysis
    center = (edges[1:] + edges[:-1])*0.5

    # import results of the first point
    par_val, d_par_plus, d_par_minus = np.loadtxt(main_dir+"\\Results\\parameters_values.txt", unpack = True)

    pdf = weighted_log_normal(xx, par_val)
    w_normal_1 = par_val[0] * gauss(xx, par_val[1], par_val[2])
    w_normal_2 = (1-par_val[0]) * gauss(xx, par_val[3], par_val[4])

    # Second point of the exercise: classificate a GRB
    log_T = np.log(2.0)

    w = par_val[0]
    normal_1_at_T = gauss(log_T, par_val[1], par_val[2])
    normal_2_at_T = gauss(log_T, par_val[3], par_val[4])
    pdf_at_T = weighted_log_normal(log_T, par_val)
    
    prob_short = w * normal_1_at_T/pdf_at_T
    prob_long = (1-w) * normal_2_at_T/pdf_at_T
    print(f"Probability that is short: {prob_short}")
    print(f"Probability that is long: {prob_long}")
    print(f"Probability that is long or short: {prob_short+prob_long}")

    plt.figure("Model with GRB to classificate")
    plt.plot(xx, pdf, 'C0', label = "Model")
    plt.axvline(log_T, color = 'r', label = "GRB170817A", linestyle = 'dashed')
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.show()