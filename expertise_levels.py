#!python3
"""
Utilities for computing random expertise levels.

"""


import numpy as np
from scipy import stats

MIN_PROBABILITY=0.501
MAX_PROBABILITY=0.999

def fixed(mean:float, size:int):
	return np.array([mean]*size)

class TruncNorm:
    def __init__(self, min_value:float, max_value:float):
        """
        Initialize a Truncated Normal distribution object, with the given minimum and maximum values 
        (the mean and std are given as arguments to functions below).
        """
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, mean:float, std:float, size:int):
        """
        Draw "size" random expertise levels.
        Each level is a probability in [0.5,1],
        drawn from a truncated-normal distribution with the given `mean` and `std`.
        :return: an array of `size` random expretise levels, sorted from high to low.

        >>> MIN_PROBABILITY=0.501
        >>> MAX_PROBABILITY=0.999
        >>> t = TruncNorm(MIN_PROBABILITY,MAX_PROBABILITY)
        >>> a = t(mean=0.7, std=0.01, size=100)
        >>> np.round(sum(a)/len(a),2)
        0.7
        >>> sum(a<MIN_PROBABILITY)
        0
        >>> sum(a>MAX_PROBABILITY)
        0
        """
        scale = std
        loc = mean

        a = (self.min_value - loc) / scale     # MIN_PROBABILITY = a*scale + loc
        b = (self.max_value - loc) / scale     # MAX_PROBABILITY = b*scale + loc
        values = stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)
        return -np.sort(-values)
    
    def true_mean(self, mean:float, std:float):
        """
        Compute the true mean of the truncated-normal distribution with the given mean and std,
        between MIN_PROBABILITY and MAX_PROBABILITY.

        >>> MIN_PROBABILITY=0.501
        >>> MAX_PROBABILITY=0.999
        >>> t = TruncNorm(MIN_PROBABILITY,MAX_PROBABILITY)
        
        >>> np.round(t.true_mean(mean=0.5, std=0.02),3)
        0.517
        >>> np.round(t.true_mean(mean=0.5, std=0.03),3)
        0.525
        >>> np.round(t.true_mean(mean=0.75, std=0.03),3)
        0.75
        >>> np.round(t.true_mean(mean=1, std=0.03),3)
        0.975
        """
        scale = std
        loc = mean
        a = (self.min_value - loc) / scale     # MIN_PROBABILITY = a*scale + loc
        b = (self.max_value - loc) / scale     # MAX_PROBABILITY = b*scale + loc
        try:
            return stats.truncnorm.mean(a,b)*scale + loc
        except FloatingPointError as err:
            raise ValueError(f"Error when computing true_mean with min_value={self.min_value}, max_value={self.max_value}, a={a}, b={b}, mean={mean}, std={std}\n\t{err}")

    def true_std(self, mean:float, std:float):
        """
        Compute the true standard deviation of the truncated-normal distribution with the given mean and std,
        between MIN_PROBABILITY and MAX_PROBABILITY.

        >>> MIN_PROBABILITY=0.501
        >>> MAX_PROBABILITY=0.999
        >>> t = TruncNorm(MIN_PROBABILITY,MAX_PROBABILITY)
        
        >>> np.round(t.true_std(mean=0.5, std=0.02),3)
        0.012
        >>> np.round(t.true_std(mean=0.5, std=0.03),3)
        0.018
        >>> np.round(t.true_std(mean=0.75, std=0.03),3)
        0.03
        >>> np.round(t.true_std(mean=1, std=0.03),3)
        0.018
        """
        scale = std
        loc = mean
        a = (self.min_value - loc) / scale     # MIN_PROBABILITY = a*scale + loc
        b = (self.max_value - loc) / scale     # MAX_PROBABILITY = b*scale + loc
        return stats.truncnorm.std(a,b)*scale
    
    def __str__(self):
        return f"TruncNorm[{self.min_value},{self.max_value}]"


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
    import doctest, numpy as np
    np.set_printoptions(legacy="1.25")
    print(doctest.testmod())

    t = TruncNorm(0.001, 0.999)
    print(t.true_mean(0.8,0.02))

    # print(stats.truncnorm.mean(a=-0.25, b=0.35, loc=0, scale=2))
    # print(stats.truncnorm.std(a=-0.25, b=0.25, loc=0, scale=1))
    # print(stats.truncnorm.std(a=-0.15, b=0.35, loc=0, scale=1))
    # print(stats.truncnorm.std(a=-5, b=5, loc=0, scale=1))
    

    # print("truncnorm: ",truncnorm(mean=0.6, std=0.1, size=11))
    # print("beta     : ",beta(mean=0.6, std=0.1, size=11))
    # print("uniform  : ",uniform(mean=0.6, std=0.1/np.sqrt(3), size=11))

    # print(beta(mean=3/4, std=1/np.sqrt(48), size=11))   # equivalent to uniform (std=0.14433)
    # print(beta(mean=0.55, std=0.14, size=11))   # almost equivalent to uniform 
    # print(beta(mean=0.75, std=0.14, size=11))   # almost equivalent to uniform 
    # print(beta(mean=0.95, std=0.14, size=11))   
    # print(truncnorm(mean=0.6, std=0, size=11))   # Division by zero error

    # import matplotlib.pyplot as plt
    # plt.hist(beta(8/14, 1.1/14, 10000), 100)    # ~ alpha=5, beta=2
    # plt.show()
    # print(beta(10/14, 1.1/14, 1))

