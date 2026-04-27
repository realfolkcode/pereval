import numpy as np
import scipy.stats as stats


def calculate_ci(values):
    ci = 0.0
    std_dev = np.std(values, ddof=1)
    n = len(values)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    std_err = std_dev / np.sqrt(n)
    ci = t_value * std_err
    return ci
