# Python script for points 2 and 3: classification of a GRB and probabilities of short and long

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
    from scipy.interpolate import interp1d
    import os

    rng = np.random.default_rng(1313) # initialize seed for reproducibility

    main_dir = os.path.dirname(os.path.realpath(__file__))

    # import results of the first point
    xx = np.linspace(-4,7,256)
    par_val, d_par_plus, d_par_minus = np.loadtxt(main_dir+"\\Results\\1a\\parameters_values.txt", unpack = True)
    pdf, w_normal_1, w_normal_2 = np.loadtxt(main_dir+"\\Results\\1a\\model_values.txt", unpack = True)


    # Second point of the exercise: classifying a GRB
    log_T = np.log(2.0)

    w_normal_1_at_T = interp1d(xx, w_normal_1)(log_T)
    w_normal_2_at_T = interp1d(xx, w_normal_2)(log_T)
    pdf_at_T = interp1d(xx, pdf)(log_T)
    
    prob_short_at_T = w_normal_1_at_T/pdf_at_T
    prob_long_at_T = w_normal_2_at_T/pdf_at_T
    print(f"Probability that is short: {prob_short_at_T:.3f}")
    print(f"Probability that is long: {prob_long_at_T:.3f}")
    print(f"Probability that is long or short: {prob_short_at_T+prob_long_at_T}")

    fig_class = plt.figure("Model with GRB to classifying")
    ax = fig_class.add_subplot(111)
    ax.axvline(log_T, color = 'k', label = "GRB170817A", linestyle = 'dashed', zorder=4)
    ax.plot(xx, pdf, 'r', label = "Model")
    ax.plot(xx, w_normal_1, 'g', label = "Norm 1", alpha = 0.75)
    ax.plot(xx, w_normal_2, color = 'darkorange', label = "Norm 2", alpha = 0.75)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Probability")
    ax.set_ylim(bottom=0)
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    plt.legend()

    # Point 3: decide a figure of merit for the transition between long and short GRBs
    prob_short = w_normal_1/pdf
    prob_long = w_normal_2/pdf

    # find the transition point
    log_T_trans_plus = xx[prob_short < prob_long][0]
    log_T_trans_minus = xx[prob_short > prob_long][-1]
    log_T_trans = np.mean([log_T_trans_minus, log_T_trans_plus])
    print(f"The transition point is at logT = {log_T_trans:.2f}, so for T = {np.exp(log_T_trans):.2f} s")

    # transition with delta as fraction? How much difference we want?
    Tr = 10                             # threshold for transition
    delta = prob_short/prob_long        # fraction of probabilities

    begin_p = xx[delta<Tr][0]                    # first x with delta<Tr
    begin_m = xx[xx<begin_p][-1]                # last x with delta>Tr
    begin = np.mean([begin_p, begin_m])
    print(f"The transition begins at logT = {begin:.2f}, so T = {np.exp(begin):.2f} s")

    end_m = xx[delta>1/Tr][-1]                   # last point with delta>1/Tr
    end_p = xx[xx>end_m][0]                     # first point with delta>1/Tr
    end = np.mean([end_p, end_m])
    print(f"The transition ends at logT = {end:.2f}, so T = {np.exp(end):.2f} s")


    fig_prob = plt.figure("Probabilities for the two class of GRB")
    ax = fig_prob.add_subplot(111)
    ax.plot(xx, prob_short, 'g', label = "$P(S|TI')$")
    ax.plot(xx, prob_long, color = 'darkorange', label = "$P(L|TI')$")   
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_xlim([-4,7])
    ax.set_ylabel("Probability")
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    leg_loc = (0.7,0.6)
    ax.legend(loc='upper left', bbox_to_anchor = leg_loc)
    
    # end of point, show and or save images

    plt.show()
    exit()

    path = main_dir+"\\Results\\2_3\\"
    # Check if the results dir exists
    if not os.path.exists(path):
        os.mkdir(path)

    fig_class.savefig(path+"GRB_to_class.png", dpi = 600)
    fig_prob.savefig(path+"0_Prob_of_class.png", dpi = 600)
    ax.axvline(log_T_trans, color = 'r', label = "Threshold", linestyle = 'dashed')
    ax.legend(loc='upper left', bbox_to_anchor = leg_loc)
    fig_prob.savefig(path+"1_Prob_of_class.png", dpi = 600)
    ax.axvline(begin, color = 'b', linestyle = 'dashdot', alpha = 0.5)
    ax.axvline(end, color = 'b', label = "Transition", linestyle = 'dashdot', alpha = 0.5) 
    ax.legend(loc='upper left', bbox_to_anchor = leg_loc)
    fig_prob.savefig(path+"2_Prob_of_class.png", dpi = 600)