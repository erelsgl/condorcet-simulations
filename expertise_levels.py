#!python3
"""
Utilities for computing random expertise levels.

"""


import numpy as np
from scipy.stats import truncnorm, beta


def fixed_expertise_levels(mean:float, size:int):
	return np.array([mean]*size)

MIN_PROBABILITY=0.501
MAX_PROBABILITY=0.999

def truncnorm_expertise_levels(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is a probability in [0.5,1],
    drawn from a truncated-normal distribution with the given `mean` and `std`.
    :return: an array of `size` random expretise levels, sorted from high to low.

    >>> a = truncnorm_expertise_levels(mean=0.7, std=0.01, size=100)
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
    return -np.sort(-truncnorm.rvs(a, b, loc=loc, scale=scale, size=size))



def beta_expertise_levels(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is a probability in [0.5,1],
    drawn from a beta distribution with the given `mean` and `std`.
    :return: an array of `size` random expretise levels, sorted from high to low.

    >>> a = beta_expertise_levels(mean=0.7, std=0.01, size=100)
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

    # print(f"a={a} b={b}")

    try:
        values = beta.rvs(a, b, loc=loc, scale=scale, size=size)
        return -np.sort(-values)
    except ValueError as err:
        print(f"mean={mean}, std={std}, loc={loc}, scale={scale}, mean1={mean1}, std1={std1}, a={a}, b={b}")
        raise err




if __name__ == "__main__":
    import doctest
    # (failures,tests) = doctest.testmod(report=True)
    # print ("{} failures, {} tests".format(failures,tests))

    print(truncnorm_expertise_levels(mean=0.6, std=0.1, size=11))
    print(beta_expertise_levels(mean=0.6, std=0.1, size=11))
    print(beta_expertise_levels(mean=3/4, std=1/np.sqrt(48), size=11))   # equivalent to uniform (std=0.14433)
    print(beta_expertise_levels(mean=0.55, std=0.14, size=11))   # almost equivalent to uniform 
    print(beta_expertise_levels(mean=0.75, std=0.14, size=11))   # almost equivalent to uniform 
    print(beta_expertise_levels(mean=0.95, std=0.14, size=11))   
    # print(truncnorm_expertise_levels(mean=0.6, std=0, size=11))   # Division by zero error


