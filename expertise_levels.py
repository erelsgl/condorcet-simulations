#!python3
"""
Utilities for computing random expertise levels.

"""


import numpy as np
from scipy import stats


def fixed(mean:float, size:int):
	return np.array([mean]*size)

MIN_PROBABILITY=0.501
MAX_PROBABILITY=0.999

def truncnorm(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is a probability in [0.5,1],
    drawn from a truncated-normal distribution with the given `mean` and `std`.
    :return: an array of `size` random expretise levels, sorted from high to low.

    >>> a = truncnorm(mean=0.7, std=0.01, size=100)
    >>> np.round(sum(a)/len(a),2)
    0.7
    >>> sum(a<MIN_PROBABILITY)
    0
    >>> sum(a>MAX_PROBABILITY)
    0
    """
    scale = std
    loc = mean
    
    a = (MIN_PROBABILITY - loc) / scale     # MIN_PROBABILITY = a*scale + loc
    b = (MAX_PROBABILITY - loc) / scale     # MAX_PROBABILITY = b*scale + loc
    values = stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)
    return -np.sort(-values)


def truncnorm_true_mean(mean:float, std:float):
    """
    Compute the true mean of the truncated-normal distribution with the given mean and std,
    between MIN_PROBABILITY and MAX_PROBABILITY.

    >>> np.round(truncnorm_true_mean(mean=0.5, std=0.02),3)
    0.517
    >>> np.round(truncnorm_true_mean(mean=0.5, std=0.03),3)
    0.525
    >>> np.round(truncnorm_true_mean(mean=0.75, std=0.03),3)
    0.75
    >>> np.round(truncnorm_true_mean(mean=1, std=0.03),3)
    0.975
    """
    scale = std
    loc = mean
    a = (MIN_PROBABILITY - loc) / scale     # MIN_PROBABILITY = a*scale + loc
    b = (MAX_PROBABILITY - loc) / scale     # MAX_PROBABILITY = b*scale + loc
    return stats.truncnorm.mean(a,b)*scale + loc

truncnorm.true_mean = truncnorm_true_mean


def beta(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is a probability in [0.5,1],
    drawn from a beta distribution with the given `mean` and `std`.
    :return: an array of `size` random expretise levels, sorted from high to low.

    >>> a = beta(mean=0.7, std=0.01, size=100)
    >>> np.round(sum(a)/len(a),2)
    0.7
    >>> sum(a<MIN_PROBABILITY)
    0
    >>> sum(a>MAX_PROBABILITY)
    0
    """
    loc = MIN_PROBABILITY           # MIN_PROBABILITY = 0*scale + loc
    scale = MAX_PROBABILITY - loc   # MAX_PROBABILITY = 1*scale + loc

    # from here: https://stats.stackexchange.com/a/12239/10760
    # a = mean^2 * [ (1-mean)/std^2 - (1/mean) ]
    # b = a*(1/mean - 1)

    # Compute the mean of the standard beta (which is in [0,1]):
    mean1 = (mean-loc)/scale    # mean = mean1*scale + loc
    std1  = std/scale           # std  = std1*scale
    a = (mean1**2) * ( (1-mean1)/(std1**2) - (1/mean1) )
    b = a*(1/mean1 - 1)

    print(f"14a={a} b={b}")

    try:
        values = stats.beta.rvs(a, b, loc=loc, scale=scale, size=size)
        return -np.sort(-values)
    except ValueError as err:
        print(f"mean={mean}, std={std}, loc={loc}, scale={scale}, mean1={mean1}, std1={std1}, a={a}, b={b}")
        raise err



def uniform(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is drawn from a uniform distribution in [mean-std*sqrt(3), mean+std*sqrt(3)],
    which indeed has the given mean and std.
    :return: an array of `size` random expretise levels, sorted from high to low.

    >>> a = uniform(mean=0.7, std=0.01/np.sqrt(3), size=100)
    >>> np.round(sum(a)/len(a),1)
    0.7
    >>> sum(a<0.69)
    0
    >>> sum(a>0.71)
    0
    """
    # std = (2*radius)/sqrt(12) = radius/sqrt(3)
    radius = np.round(std * np.sqrt(3), 2)
    loc = mean-radius
    scale = 2*radius
    # print(f"loc={loc}, scale={scale}")
    values = stats.uniform.rvs(loc=loc, scale=scale, size=size)
    return -np.sort(-values)


if __name__ == "__main__":
    import doctest
    # print(doctest.testmod())

    # print("truncnorm: ",truncnorm(mean=0.6, std=0.1, size=11))
    # print("beta     : ",beta(mean=0.6, std=0.1, size=11))
    # print("uniform  : ",uniform(mean=0.6, std=0.1/np.sqrt(3), size=11))

    # print(beta(mean=3/4, std=1/np.sqrt(48), size=11))   # equivalent to uniform (std=0.14433)
    # print(beta(mean=0.55, std=0.14, size=11))   # almost equivalent to uniform 
    # print(beta(mean=0.75, std=0.14, size=11))   # almost equivalent to uniform 
    # print(beta(mean=0.95, std=0.14, size=11))   
    # print(truncnorm(mean=0.6, std=0, size=11))   # Division by zero error

    import matplotlib.pyplot as plt
    # plt.hist(beta(8/14, 1.1/14, 10000), 100)    # ~ alpha=5, beta=2
    # plt.show()
    print(beta(10/14, 1.1/14, 1))

