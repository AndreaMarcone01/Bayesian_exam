import numpy as np
from scipy.interpolate import interp1d

def percentile(xx, pdf, perc):
    """From the PDF find the values of xx where a certain percentile is hit.

    Args:
        xx (np.array): array of x values
        pdf (np.array): probability density function as function of xx
        perc (np.array): percentile we want to find between 0 and 1

    Returns:
        np.array: values of xx where the percentile is hit
    """
    
    # find the CDF
    area_trap = np.diff(xx) * (pdf[:-1] + pdf[1:]) /2
    cdf = np.concatenate([[0], np.cumsum(area_trap)])   # to assure first is zero
    cdf = cdf/cdf[-1]                                   # to assure last is one

    inverse_cdf = interp1d(cdf, xx)
    find_perc = inverse_cdf(perc)

    return find_perc

def errors_around_peak(xx, pdf):
    """From the PDF find the median value of xx togheter with 90% confidence interval.

    Args:
        xx (np.array): array of x values
        pdf (np.array): probability density function as function of xx

    Returns:
        tuple:
            - xx_peak(float): value of xx at the median of the pdf
            - error_plus(float): positive error, defined as when the cdf hits 95% minus xx_peak
            - error_minus(float): negative error, defined as xx_peak minus when the cdf hits 5%
    """
    perc = percentile(xx, pdf, [0.05,0.50, 0.95])
    xx_min = float(perc[0])
    xx_peak = float(perc[1])
    xx_max = float(perc[2])
    
    error_plus = xx_max - xx_peak
    error_minus = xx_peak - xx_min

    return xx_peak, error_plus, error_minus   