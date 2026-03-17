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

    # import results of the first point
    xx = np.linspace(-4,7,256)
    par_val, d_par_plus, d_par_minus = np.loadtxt(main_dir+"\\Results\\parameters_values.txt", unpack = True)

    pdf = weighted_log_normal(xx, par_val)
    w_normal_1 = par_val[0] * gauss(xx, par_val[1], par_val[2])
    w_normal_2 = (1-par_val[0]) * gauss(xx, par_val[3], par_val[4])

    # Second point of the exercise: classifying a GRB
    log_T = np.log(2.0)

    w = par_val[0]
    normal_1_at_T = gauss(log_T, par_val[1], par_val[2])
    normal_2_at_T = gauss(log_T, par_val[3], par_val[4])
    pdf_at_T = weighted_log_normal(log_T, par_val)
    
    prob_short_at_T = w * normal_1_at_T/pdf_at_T
    prob_long_at_T = (1-w) * normal_2_at_T/pdf_at_T
    print(f"Probability that is short: {prob_short_at_T:.3f}")
    print(f"Probability that is long: {prob_long_at_T:.3f}")
    print(f"Probability that is long or short: {prob_short_at_T+prob_long_at_T}")

    plt.figure("Model with GRB to classifying")
    plt.plot(xx, pdf, 'C0', label = "Model")
    plt.axvline(log_T, color = 'r', label = "GRB170817A", linestyle = 'dashed')
    plt.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.5)
    plt.plot(xx, w_normal_2, color = 'orange', label = "Norm 2", alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.savefig(main_dir+"\\Results\\GRB_to_class.png")

    # Point 3: decide a figure of merit for the transition between long and short GRBs
    prob_short = w_normal_1/pdf
    prob_long = w_normal_2/pdf
    delta = np.abs(prob_short-prob_long)

    # find the transition point
    log_T_trans_plus = xx[prob_short < prob_long][0]
    log_T_trans_minus = xx[prob_short > prob_long][-1]
    log_T_trans = np.mean([log_T_trans_minus, log_T_trans_plus])
    print(f"The transition point is at logT = {log_T_trans:.2f}, so for T = {np.exp(log_T_trans):.2f} s")

    begin_p = xx[delta<0.9][0]
    delta_p = delta[xx<begin_p]
    begin_m = xx[xx<begin_p][delta_p>0.9][-1]
    begin = np.mean([begin_p, begin_m])
    print(f"The transition begins at logT = {begin:.2f}, so T = {np.exp(begin):.2f} s")

    end_m = xx[delta<0.9][-1]
    delta_m = delta[xx>end_m]
    end_p = xx[xx>end_m][delta_m>0.9][0]
    end = np.mean([end_p, end_m])
    print(f"The transition ends at logT = {end:.2f}, so T = {np.exp(end):.2f} s")

    plt.figure("Probabilities for the two class of GRB")
    plt.plot(xx, prob_short, 'g', label = "P short")
    plt.plot(xx, prob_long, color = 'orange', label = "P long")
    plt.axvline(log_T_trans, color = 'r', label = "Threshold", linestyle = 'dashed')
    plt.axvline(begin, color = 'b', linestyle = 'dashed', alpha = 0.5)
    plt.axvline(end, color = 'b', label = "Transition", linestyle = 'dashed', alpha = 0.5)
    plt.xlabel("$\\log(T_{90})$")
    plt.xlim([-4,7])
    plt.ylabel("Probability")
    plt.grid(linestyle = 'dashed')
    plt.legend()
    plt.savefig(main_dir+"\\Results\\Prob_of_class.png")
    
    plt.show()