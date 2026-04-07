import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    main_dir = os.path.dirname(os.path.realpath(__file__))
    xx = np.linspace(-4,7,256)

    # check differences between the point 1a and 1b
    par, par_plus, par_minus = np.loadtxt(main_dir+"\\Results\\1a\\parameters_values.txt", unpack=True)
    e_par, e_par_plus, e_par_minus = np.loadtxt(main_dir+"\\Results\\1b\\Err_parameters_values.txt", unpack=True)
    name_l = ["$w$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$"]
    
    fig0 = plt.figure("Confront parameters", figsize = (6,6))
    for i in range(len(par)):
        if i > 0:
            ax = fig0.add_subplot(3, 2, i+2)
        else:
            ax = fig0.add_subplot(3, 2, i+1)
        
        errors = [[par_minus[i]], [par_plus[i]]]
        ax.errorbar(0, par[i], yerr=[[par_minus[i]], [par_plus[i]]], color = 'c', fmt= 'o')
        ax.errorbar(1, e_par[i], yerr=[[e_par_minus[i]], [e_par_plus[i]]], color = 'm', fmt= 'o')
        ax.set_xticks([0,1], ['No errors', 'With errors'])
        ax.set_ylabel(name_l[i])
    plt.tight_layout()

    pdf, norm_1, norm_2 = np.loadtxt(main_dir+"\\Results\\1a\\model_values.txt", unpack=True)
    e_pdf, e_norm_1, e_norm_2 = np.loadtxt(main_dir+"\\Results\\1b\\Err_model_values.txt", unpack=True)
    
    fig1 = plt.figure("Data and model")
    ax = fig1.add_subplot(111)
    ax.plot(xx, pdf, color = 'r', label = 'Without errors', linestyle='--', zorder=2)
    ax.plot(xx, e_pdf, 'r', label = "With errors", alpha = 0.5, zorder=3)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_ylabel("Probability")
    ax.set_ylim(0,0.30)
    ax.legend(loc='upper left')
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)
    """
    ax_diff = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_diff.tick_params(axis="x", labelbottom=False)
    ax_diff.plot(xx, pdf-e_pdf, color = 'C0', label = 'Difference')
    ax_diff.set_ylabel("Difference")
    ax_diff.grid(linestyle = 'dashed')
    ax_diff.set_axisbelow(True)
    plt.legend()
    plt.tight_layout()
    """

    fig2 = plt.figure("Norm 1")
    ax = fig2.add_subplot(111)
    ax.plot(xx, norm_1, color = 'g', label = 'Without errors', linestyle='--', zorder=2)
    ax.plot(xx, e_norm_1, 'g', label = "With errors", alpha = 0.5, zorder=3)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_xlim(-4,4)
    ax.set_ylabel("Probability")
    ax.set_ylim(0,0.125)
    ax.legend(loc='upper left')
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)

    fig3 = plt.figure("Norm 2")
    ax = fig3.add_subplot(111)
    ax.plot(xx, norm_2, color = 'darkorange', label = 'Without errors', linestyle='--', zorder=2)
    ax.plot(xx, e_norm_2, 'darkorange', label = "With errors", alpha = 0.5, zorder=3)
    ax.set_xlabel("$\\log(T_{90})$")
    ax.set_xlim(0,7)
    ax.set_ylabel("Probability")
    ax.set_ylim(0,0.30)
    ax.legend(loc='upper left')
    ax.grid(linestyle = 'dashed')
    ax.set_axisbelow(True)

    # end of script: save the images or show and exit
    
    plt.show()
    exit()

    path = main_dir+"\\Results\\1c"
    # Check if the results dir exists
    if not os.path.exists(path):
        os.mkdir(path)

    fig1.savefig(path+"\\Total_model_conf.png", dpi = 600)
    fig2.savefig(path+"\\Norm_1_conf.png", dpi = 600)
    fig3.savefig(path+"\\Norm_2_conf.png", dpi = 600)