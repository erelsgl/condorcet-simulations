#!python3


"""
A utility for performing simulation experiments on voting situations.

There are voters with different expertise level,
and it is required to decide whether the majority is correct,
or - alternatively - whether it is good to give the minority a decisive power.

Since:  2020-04
Author: Erel Segal-Halevi

"""


from tee_table.tee_table import TeeTable
from collections import OrderedDict
from scipy.stats import truncnorm
import numpy as np

import logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


TABLE_COLUMNS = ["iterations","voters", "mean", "std",
                 "minority_decisive", "majority_correct"]

def random_expertise_levels(mean:float, std:float, size:int):
    """
    Draw "size" random expertise levels.
    Each level is a probability in [0.5,1],
    drawn from a truncated-normal distribution with the given `mean` and `std`.
    :return: an array of `size` random expretise levels, sorted from high to low.
    """
    scale = std
    loc = mean
    lower = 0.5
    upper = 1
    # a*scale + loc = lower
    # b*scale + loc = upper
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return -np.sort(-truncnorm.rvs(a, b, loc=loc, scale=scale, size=size))

def logodds(expertise_level: float):
    return np.log( expertise_level  / (1-expertise_level))

def is_minority_decisive(sorted_expertise_levels:list, minority_size:int=None)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is decisive in the optimal decision rule.
    >>> is_minority_decisive(np.array([0.8]))
    False
    >>> is_minority_decisive(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisive(np.array([0.9, 0.6, 0.6]))
    True
    """
    if minority_size is None:
        committee_size = len(sorted_expertise_levels)
        minority_size = int( (committee_size-1)/2 )
    minority_group = sorted_expertise_levels[0:minority_size]
    majority_group = sorted_expertise_levels[minority_size:]
    minority_weight = sum(logodds(minority_group))
    majority_weight = sum(logodds(majority_group))
    return minority_weight > majority_weight


def experiment(results_csv_file:str, num_of_iterations:int, num_of_voters:list, expertise_mean:float, expertise_std:float):
    """
    Run an experiment with voters of different expertise level.

    There are `num_of_voters` voters drawn at random.
    The expertise of a voter is the probability that the voter is correct.
    It is a truncated-normal random variable, with mean  `expertise_mean`  and std  `expertise_std,
       truncated to be in the range [0.5 , 1].

    :param results_csv_file:   where to store the results.
    :param num_of_iterations:  how many times to run the experiment.
    :param num_of_voters:     how many voters are there in the committee. Should be an odd integer.
    :param expertise_mean:     what is the mean expertise-level  (should be between 0.5 and 1).
    :param expertise_std:      what is the std of the expertise level.
    """
    results_table = TeeTable(TABLE_COLUMNS, results_csv_file)

    minority_decisive_sum = 0
    minority_size = int((num_of_voters-1)/2)
    for _ in range(num_of_iterations):
        expertise_levels = random_expertise_levels(expertise_mean, expertise_std, num_of_voters)
        minority_decisive_sum += is_minority_decisive(expertise_levels, minority_size=minority_size)

    results_table.add(OrderedDict((
        ("iterations", num_of_iterations),
        ("voters", num_of_voters),
        ("mean", expertise_mean),
        ("std", expertise_std),
        ("minority_decisive", minority_decisive_sum/num_of_iterations),
        ("majority_correct", 1.0),
    )))
    results_table.done()


if __name__ == "__main__":
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print ("{} failures, {} tests".format(failures,tests))

    results_file="results/voting-1.csv"
    experiment(results_file, num_of_iterations=100, num_of_voters=9, expertise_mean=0.7, expertise_std=0.1)
