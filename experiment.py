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
import pandas
import matplotlib.pyplot as plt

import logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


TABLE_COLUMNS = ["iterations","voters", "mean", "std",
                 "minority_decisive", "minority_tyrannic", "expert_decisive",
                 "majority_correct", "expert_correct"]

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
    "decisive" means that, if ALL members of minority agree, then their opinion is accepted.
    >>> is_minority_decisive(np.array([0.8]))
    False
    >>> is_minority_decisive(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisive(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_minority_decisive(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_minority_decisive(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    True
    >>> is_minority_decisive(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), minority_size=1)
    False
    >>> is_minority_decisive(np.array([0.9, 0.9, 0.6, 0.6, 0.6]), minority_size=3)
    True
    """
    if minority_size is None:
        committee_size = len(sorted_expertise_levels)
        minority_size = int( (committee_size-1)/2 )
    weights = logodds(sorted_expertise_levels)
    half_total_weight = sum(weights)/2
    minority_weight = sum(weights[0:minority_size])
    return minority_weight > half_total_weight


def is_minority_tyrannic(sorted_expertise_levels:list)->bool:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :return: whether the minority of experts is tyrannic in the optimal decision rule.
    "tyrannic" means that the decision is accepted only by a vote within the minority, ignoring the majority altogether.
    NOTE: that "tyrannic" implies "decisive", but not vice-versa.
    >>> is_minority_tyrannic(np.array([0.8]))
    False
    >>> is_minority_tyrannic(np.array([0.8, 0.8, 0.8]))
    False
    >>> is_minority_tyrannic(np.array([0.9, 0.6, 0.6]))
    True
    >>> is_minority_tyrannic(np.array([0.8, 0.8, 0.8, 0.8, 0.8]))
    False
    >>> is_minority_tyrannic(np.array([0.9, 0.9, 0.6, 0.6, 0.6]))
    True
    """
    committee_size = len(sorted_expertise_levels)
    weights = logodds(sorted_expertise_levels)
    half_total_weight = sum(weights)/2
    if committee_size <= 5:  # minority size = 2.
        rule_10_000_optimal = weights[0] > half_total_weight
        return rule_10_000_optimal
    elif committee_size <= 7:  # minority size = 3.
        rule_100_0000_optimal = weights[0] > half_total_weight
        rule_111_0000_optimal = weights[1]+weights[2] > half_total_weight
        return rule_100_0000_optimal or rule_111_0000_optimal
    elif committee_size <= 9:  # minority size = 4.
        # [(1,0,0,0),(0,0,0,0,0)] and [(1,1,1,0),(0,0,0,0,0)] and [(2,1,1,1),(0,0,0,0,0)].
        rule_1000_00000_optimal = weights[0] > half_total_weight
        rule_1110_00000_optimal = weights[1] + weights[2] > half_total_weight
        rule_2111_00000_optimal = (weights[0] + weights[3] > half_total_weight) \
                                  or (weights[1] + weights[2] + weights[3] > half_total_weight)
        return rule_1000_00000_optimal or rule_1110_00000_optimal or rule_2111_00000_optimal
    elif committee_size <= 11:  # minority size = 5.
        # [(1,0,0,0),(0,0,0,0,0)] and [(1,1,1,0),(0,0,0,0,0)] and [(2,1,1,1),(0,0,0,0,0)].
        rule_10000_000000_optimal = weights[0] > half_total_weight
        rule_11100_000000_optimal = weights[1] + weights[2] > half_total_weight
        rule_11111_000000_optimal = weights[2] + weights[3] + weights[4] > half_total_weight
        rule_21110_000000_optimal = (weights[0] + weights[3] > half_total_weight) \
                                    or (weights[1] + weights[2] + weights[3] > half_total_weight)
        rule_22111_000000_optimal = weights[2] + weights[3] + weights[4] > half_total_weight
        rule_32211_000000_optimal = weights[2] + weights[3] + weights[4] > half_total_weight
        rule_31111_000000_optimal = weights[2] + weights[3] + weights[4] > half_total_weight
        return rule_10000_000000_optimal or rule_11100_000000_optimal or rule_11111_000000_optimal \
               or rule_21110_000000_optimal or rule_22111_000000_optimal or rule_32211_000000_optimal or rule_31111_000000_optimal
    else:
        raise ValueError("Committee sizes larger than 11 are not supported")


def fraction_majority_correct(sorted_expertise_levels:list, num_of_decisions:int)->float:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the majority rule accepts the correct decision.
    >>> f08 = fraction_majority_correct(np.array([0.8]), 1000)
    >>> np.abs(f08-0.8) < 0.05
    True
    >>> f08 = fraction_majority_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f08-0.896) < 0.05
    True
    >>> f08 = fraction_majority_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f08-.792) < 0.05
    True
    """
    committee_size = len(sorted_expertise_levels)
    num_majority_correct = 0
    for _ in range(num_of_decisions):
        num_correct = 0
        for level in sorted_expertise_levels:
            is_expert_correct = np.random.random() < level
            num_correct += is_expert_correct
        is_majority_correct = 2*num_correct >= committee_size
        num_majority_correct += is_majority_correct
    return num_majority_correct / num_of_decisions


def fraction_expert_correct(sorted_expertise_levels:list, num_of_decisions:int)->float:
    """
    :param sorted_expertise_levels: a list of expertise-levels (numbers in [0.5,1]), sorted from high to low.
    :param num of decisions to make
    :return: the empirical fraction of decisions in which the expert rule accepts the correct decision.
    >>> f08 = fraction_expert_correct(np.array([0.8]), 1000)
    >>> np.abs(f08-0.8) < 0.05
    True
    >>> f08 = fraction_expert_correct(np.array([0.8, 0.8, 0.8]), 1000)
    >>> np.abs(f08-0.8) < 0.05
    True
    >>> f08 = fraction_expert_correct(np.array([0.9, 0.6, 0.6]), 1000)
    >>> np.abs(f08-0.9) < 0.05
    True
    """
    num_expert_correct = 0
    for _ in range(num_of_decisions):
        level = sorted_expertise_levels[0]
        is_expert_correct = np.random.random() < level
        num_expert_correct += is_expert_correct
    return num_expert_correct / num_of_decisions


def create_results(results_csv_file:str, num_of_iterations:int, num_of_voterss:list, expertise_means:list, expertise_stds:list, num_of_decisions:int=2):
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

    for num_of_voters in num_of_voterss:
        for expertise_mean in expertise_means:
            for expertise_std in expertise_stds:
                minority_decisive_sum = 0
                minority_tyrannic_sum = 0
                expert_decisive_sum = 0
                majority_correct_sum = 0
                expert_correct_sum = 0
                minority_size = int((num_of_voters-1)/2)
                for _ in range(num_of_iterations):
                    expertise_levels = random_expertise_levels(expertise_mean, expertise_std, num_of_voters)
                    # minority_decisive_sum += is_minority_decisive(expertise_levels, minority_size=minority_size)
                    minority_tyrannic_sum += is_minority_tyrannic(expertise_levels)
                    # expert_decisive_sum += is_minority_decisive(expertise_levels, minority_size=1)
                    # majority_correct_sum  += fraction_majority_correct(expertise_levels, num_of_decisions=num_of_decisions)
                    # expert_correct_sum  += fraction_expert_correct(expertise_levels, num_of_decisions=num_of_decisions)

                results_table.add(OrderedDict((
                    ("iterations", num_of_iterations),
                    ("voters", num_of_voters),
                    ("mean", expertise_mean),
                    ("std", expertise_std),
                    ("minority_decisive", minority_decisive_sum/num_of_iterations),
                    ("minority_tyrannic", minority_tyrannic_sum/num_of_iterations),
                    ("expert_decisive", expert_decisive_sum/num_of_iterations),
                    ("majority_correct", majority_correct_sum/num_of_iterations),
                    ("expert_correct", expert_correct_sum/num_of_iterations),
                )))
    results_table.done()



titleFontSize = 12
legendFontSize = 8
axesFontSize = 10
markerSize=12
style="g-o"

def plot_vs_std(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure()
    results = pandas.read_csv(results_csv_file)

    for index,num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results.loc[results['voters']==num_of_voters]
        ax = plt.subplot(2, 2, index+1)
        ax.set_title('{} voters'.format(num_of_voters),
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_mean in expertise_means:
            results_for_mean = results_for_voters.loc[results_for_voters['mean']==expertise_mean]
            x_values = results_for_mean['std']
            y_values = results_for_mean[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="mean={}".format(expertise_mean))
            if line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        ax.legend(prop={'size': legendFontSize}, loc='best')

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    plt.draw()


def plot_vs_mean(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure()
    results = pandas.read_csv(results_csv_file)

    for index,num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results.loc[results['voters']==num_of_voters]
        ax = plt.subplot(2, 2, index+1)
        ax.set_title('{} voters'.format(num_of_voters),
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_std in expertise_stds:
            results_for_std = results_for_voters.loc[results_for_voters['std']==expertise_std]
            x_values = results_for_std['mean']
            y_values = results_for_std[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="std={}".format(expertise_std))
            if line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        ax.legend(prop={'size': legendFontSize}, loc='best')

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    plt.draw()


if __name__ == "__main__":
    # import doctest
    # (failures,tests) = doctest.testmod(report=True)
    # print ("{} failures, {} tests".format(failures,tests))

    num_of_iterations = 1000
    num_of_voterss  = [5, 7, 9, 11]
    expertise_means = [.55, .6, .65, .7, .75, .8, .85, .9, .95]
    expertise_stds  = np.arange(start=0.002, stop=0.2, step=0.002)

    results_file="results/voting-1000iters-tyrannic.csv"
    create_results(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)
    plot_vs_mean(results_file, "minority_tyrannic", num_of_voterss, expertise_means, expertise_stds=[0.002, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05], line_at_half=True)
    plot_vs_std(results_file, "minority_tyrannic", num_of_voterss, expertise_means, expertise_stds, line_at_half=True)

    plt.show()
