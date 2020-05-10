#!python3


"""
A utility for performing simulation experiments on decision rules.

There are voters with different expertise level,
and it is required to decide whether the majority is correct,
or - alternatively - whether it is good to give the minority a decisive power.

Since:  2020-04
Author: Erel Segal-Halevi
"""


from tee_table.tee_table import TeeTable
from collections import OrderedDict
import pandas
import matplotlib.pyplot as plt

import numpy as np
from query_decision_rules import *
from random_expertise_levels import random_expertise_levels


import logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

from query_decision_rules import *


TABLE_COLUMNS = ["iterations","voters", "mean", "mean_bucket", "std", "std_bucket",
                 "simple_majority_optimal", "minority_decisiveness_optimal",
                 "non_tyrannic_minority_decisiveness_optimal",
                 "minority_tyranny_optimal", "expert_tyranny_optimal",
                 "minority_colluding", "majority_correct", "expert_correct"]


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
                minority_decisiveness_optimal = 0
                minority_tyranny_optimal = 0
                expert_tyranny_optimal_sum = 0
                majority_correct_sum = 0
                expert_correct_sum = 0
                minority_colluding_sum = 0
                minority_size = int((num_of_voters-1)/2)
                for _ in range(num_of_iterations):
                    expertise_levels = random_expertise_levels(expertise_mean, expertise_std, num_of_voters)
                    minority_decisiveness_optimal += is_minority_decisiveness_optimal(expertise_levels, minority_size=minority_size)
                    minority_tyranny_optimal += is_minority_tyranny_optimal(expertise_levels)
                    expert_tyranny_optimal_sum += is_minority_decisiveness_optimal(expertise_levels, minority_size=1)
                    majority_correct_sum  += fraction_majority_correct(expertise_levels, num_of_decisions=num_of_decisions)
                    expert_correct_sum  += fraction_expert_correct(expertise_levels, num_of_decisions=num_of_decisions)
                    minority_colluding_sum += fraction_minority_colluding(expertise_levels, minority_size)

                results_table.add(OrderedDict((
                    ("iterations", num_of_iterations),
                    ("voters", num_of_voters),
                    ("mean", expertise_mean),
                    ("std", expertise_std),

                    #  majority rule is optimal iff it is NOT optimal to let colluding minority decide:
                    ("simple_majority_optimal", (num_of_iterations-minority_decisiveness_optimal)/num_of_iterations),
                    ("minority_decisiveness_optimal",  minority_decisiveness_optimal / num_of_iterations),

                    #  minority decisiveness is optimal iff colluding minority is optimal but a meritocratic rule is NOT optimal:
                    ("non_tyrannic_minority_decisiveness_optimal",  (minority_decisiveness_optimal - minority_tyranny_optimal) / num_of_iterations),
                    ("minority_tyranny_optimal", minority_tyranny_optimal/num_of_iterations),
                    ("expert_tyranny_optimal", expert_tyranny_optimal_sum/num_of_iterations),

                    ("minority_colluding", minority_colluding_sum/num_of_iterations),
                    ("majority_correct", majority_correct_sum/num_of_iterations),
                    ("expert_correct", expert_correct_sum/num_of_iterations),
                )))
    results_table.done()


num_of_voterss  = [3, 5, 7, 9, 11]


expertise_means = [.5, .55, .6,
                   .7, .75, .8,
                   .9, .95, 1]

expertise_stds  = [0.02, 0.03, 0.04,
                   0.06, 0.07, 0.08,
                   0.10, 0.11, 0.12]

def create_group_results(results_csv_file:str):
    results = pandas.read_csv(results_csv_file)

    # Create buckets for the mean:
    results["mean_bucket"] = "Lower"
    results.loc[results.query('mean > 0.65').index, "mean_bucket"] = "Medium"
    results.loc[results.query('mean > 0.85').index, "mean_bucket"] = "Upper"

    # Create buckets for the std:
    results["std_bucket"] = "Lower"
    results.loc[results.query('std > 0.05').index, "std_bucket"] = "Medium"
    results.loc[results.query('std > 0.09').index, "std_bucket"] = "Upper"

    results_mean = results.groupby(['voters', 'mean_bucket', 'std_bucket']).mean()
    results_mean.drop(columns=["iterations","mean","std"], inplace=True)
    results_mean.index.names = ["voters", "mean", "std"]
    results_mean.rename(columns={
        "simple_majority_optimal": "smr",
        "minority_decisiveness_optimal": "mino-d",
        "non_tyrannic_minority_decisiveness_optimal": "n-t-m-d",
        "minority_tyranny_optimal": "mino-t",
        "expert_tyranny_optimal": "expert",
        "minority_colluding": "min-coll",
    }, inplace=True)
    results_mean.rename(columns={
        "mean_bucket":"mean",
        "std_bucket":"std",
        "simple_majority_optimal": "smr",
        "minority_decisiveness_optimal": "mino-d",
        "non_tyrannic_minority_decisiveness_optimal": "n-t-m-d",
        "minority_tyranny_optimal": "mino-t",
        "expert_tyranny_optimal": "expert",
        "minority_colluding": "min-coll",
    }, inplace=True)

    results_mean_csv_file = results_csv_file.replace(".csv", "-mean.csv")
    results_mean.to_csv(results_mean_csv_file, index=True)

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
        ax = plt.subplot(2, 3, index+1)
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
        ax = plt.subplot(2, 3, index+1)
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

    num_of_iterations = 1000
    results_file="results/{}iters-all.csv".format(num_of_iterations)

    # create_results(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)
    create_group_results(results_file)

    # for field_name in ["minority_tyranny_optimal"]:
    #     plot_vs_mean(results_file, field_name, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    #     plot_vs_std(results_file, "minority_tyranny_optimal", num_of_voterss, expertise_means, expertise_stds, line_at_half=False)

    plt.show()
