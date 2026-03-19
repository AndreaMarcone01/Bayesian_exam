import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    main_dir = os.path.dirname(os.path.realpath(__file__))

    par, par_plus, par_minus = np.loadtxt(main_dir+"\\Results\\1a\\parameters_values.txt", unpack=True)
    e_par, e_par_plus, e_par_minus = np.loadtxt(main_dir+"\\Results\\1b\\Err_parameters_values.txt", unpack=True)
    name_l = ["$w$", "$\\mu_1$", "$\\sigma_1$", "$\\mu_2$", "$\\sigma_2$"]

    
    fig = plt.figure("Confront parameters", figsize = (6,6))
    for i in range(len(par)):
        if i > 0:
            ax = fig.add_subplot(3, 2, i+2)
        else:
            ax = fig.add_subplot(3, 2, i+1)
        
        errors = [[par_minus[i]], [par_plus[i]]]
        ax.errorbar(0, par[i], yerr=[[par_minus[i]], [par_plus[i]]], color = 'c', fmt= 'o')
        ax.errorbar(1, e_par[i], yerr=[[e_par_minus[i]], [e_par_plus[i]]], color = 'm', fmt= 'o')
        ax.set_xticks([0,1], ['No errors', 'With errors'])
        ax.set_ylabel(name_l[i])

    """ax_leg = fig.add_subplot(3,2,2)
    # Create dummy artists just for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='C0', linewidth=1.5, label='Marginalised posterior samples'),
        Line2D([0], [0], color='r', linestyle='dashed', label='Prior'),
        Line2D([0], [0], color='green', linestyle='dashed', label='Median value'),
        Line2D([0], [0], color='orange', linestyle='dashed', label='Median value $\\pm\\sigma$'),
    ]
    ax_leg.legend(handles=legend_elements, loc='center')
    ax_leg.axis('off')  # Hide axes, ticks, spines and background"""
    plt.tight_layout()
    plt.show()
    