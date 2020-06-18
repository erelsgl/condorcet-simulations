#!python3


"""
A utility for performing simulation experiments on decision rules.

There are voters with different expertise level,
and it is required to decide whether the majority is correct,
or - alternatively - whether it is good to give the minority a decisive power.

Since:  2020-04
Author: Erel Segal-Halevi

Credits: https://stats.stackexchange.com/q/471426/10760
"""

from tee_table.tee_table import TeeTable
from collections import OrderedDict
import pandas, os.path
import matplotlib.pyplot as plt

from Committee import Committee

import logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

INDEX_COLUMNS = ["iterations","voters", "mean", "mean_bucket", "std", "std_bucket"]

OPTIMALITY_COLUMNS = [
                 "simple_majority_optimal", "minority_decisiveness_optimal",
                 "non_tyrannic_minority_decisiveness_optimal",
                 "minority_tyranny_optimal", "expert_tyranny_optimal", "minority_colluding"]

CORRECTNESS_COLUMNS = ["optimal_correct", "majority_correct", "expert_correct", "compromise_minority_correct", "compromose_strongmajority_correct"]

AGREEMENT_COLUMNS = ["optimal_agrees_majority", "compromise_minority_agrees_majority", "compromise_minority_agrees_optimal", "compromose_strongmajority_agrees_majority", "compromose_strongmajority_agrees_optimal"]

REPORTED_COLUMNS = OPTIMALITY_COLUMNS + CORRECTNESS_COLUMNS + AGREEMENT_COLUMNS

TABLE_COLUMNS = INDEX_COLUMNS + REPORTED_COLUMNS


def create_results(results_csv_file:str, num_of_iterations:int, num_of_voterss:list, expertise_means:list, expertise_stds:list, num_of_decisions:int=1, debug_committees=False):
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

    NOTE: num_decisions is set to 1 based on the answer by Henry:
            https://stats.stackexchange.com/a/471431/10760
    """
    results_table = TeeTable(TABLE_COLUMNS, results_csv_file)

    for num_of_voters in num_of_voterss:
        for expertise_mean in expertise_means:
            for expertise_std in expertise_stds:
                minority_decisiveness_optimal = 0
                minority_tyranny_optimal = 0
                expert_tyranny_optimal_sum = 0
                minority_colluding_sum = 0

                optimal_correct_sum = 0
                majority_correct_sum = 0
                expert_correct_sum = 0
                compromise_minority_correct_sum = 0
                compromose_strongmajority_correct_sum = 0

                optimal_agrees_majority_sum = 0
                compromise_minority_agrees_majority_sum = 0
                compromise_minority_agrees_optimal_sum = 0
                compromose_strongmajority_agrees_majority_sum = 0
                compromose_strongmajority_agrees_optimal_sum = 0

                for _ in range(num_of_iterations):
                    committee = Committee.random_expertise_levels(expertise_mean, expertise_std, num_of_voters)
                    if (debug_committees): print(committee)
                    minority_decisiveness_optimal += committee.is_minority_decisiveness_optimal()
                    minority_tyranny_optimal += committee.is_minority_tyranny_optimal()
                    expert_tyranny_optimal_sum += committee.is_minority_decisiveness_optimal(minority_size=1)
                    minority_colluding_sum += committee.fraction_minority_colluding(num_of_decisions=1)

                    for _ in range(num_of_decisions):
                        vote = committee.vote()

                        optimal_vote  = committee.optimal_weighted_rule(vote)
                        majority_vote = committee.simple_majority_rule(vote)
                        expert_vote = committee.expert_rule(vote)
                        compromise_minority_vote = committee.compromise_weights_rule(vote)
                        compromose_strongmajority_vote = committee.compromise_strongmajority_rule(vote)

                        optimal_correct_sum  += optimal_vote/num_of_decisions
                        majority_correct_sum += majority_vote/num_of_decisions
                        expert_correct_sum  += expert_vote/num_of_decisions
                        compromise_minority_correct_sum  += compromise_minority_vote/num_of_decisions
                        compromose_strongmajority_correct_sum  += compromose_strongmajority_vote/num_of_decisions

                        optimal_agrees_majority_sum += (optimal_vote==majority_vote)/num_of_decisions
                        compromise_minority_agrees_majority_sum += (compromise_minority_vote==majority_vote)/num_of_decisions
                        compromise_minority_agrees_optimal_sum += (compromise_minority_vote==optimal_vote)/num_of_decisions
                        compromose_strongmajority_agrees_majority_sum += (compromose_strongmajority_vote==majority_vote)/num_of_decisions
                        compromose_strongmajority_agrees_optimal_sum += (compromose_strongmajority_vote==optimal_vote)/num_of_decisions

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

                    ("optimal_correct", optimal_correct_sum / num_of_iterations),
                    ("majority_correct", majority_correct_sum / num_of_iterations),
                    ("expert_correct", expert_correct_sum / num_of_iterations),
                    ("compromise_minority_correct", compromise_minority_correct_sum / num_of_iterations),
                    ("compromose_strongmajority_correct", compromose_strongmajority_correct_sum / num_of_iterations),

                    ("optimal_agrees_majority", optimal_agrees_majority_sum / num_of_iterations),
                    ("compromise_minority_agrees_majority", compromise_minority_agrees_majority_sum / num_of_iterations),
                    ("compromise_minority_agrees_optimal", compromise_minority_agrees_optimal_sum / num_of_iterations),
                    ("compromose_strongmajority_agrees_majority", compromose_strongmajority_agrees_majority_sum / num_of_iterations),
                    ("compromose_strongmajority_agrees_optimal", compromose_strongmajority_agrees_optimal_sum / num_of_iterations),
                )))
    results_table.done()


def convert_probabilities_to_odds(results_csv_file:str):
    results = pandas.read_csv(results_csv_file)
    for column in REPORTED_COLUMNS:
        if column in results.columns:
            results[column] = results[column] / (1-results[column])
    results.to_csv(results_csv_file.replace(".csv", "-odds.csv"), index=True)


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

    results_mean = results.groupby(['voters', 'mean_bucket', 'std_bucket']).mean().round(3)
    results_mean.drop(columns=["iterations","mean","std"], inplace=True)
    results_mean.index.names = ["voters", "mean", "std"]

    results_mean\
        .drop(columns=CORRECTNESS_COLUMNS)\
        .drop(columns=AGREEMENT_COLUMNS)\
        .rename(columns={
            "simple_majority_optimal": "smr",
            "minority_decisiveness_optimal": "mino-d",
            "non_tyrannic_minority_decisiveness_optimal": "n-t-m-d",
            "minority_tyranny_optimal": "mino-t",
            "expert_tyranny_optimal": "expert",
            "minority_colluding": "min-coll"})\
        .to_csv(results_csv_file.replace(".csv", "-mean-optimal.csv"), index=True)
    results_mean\
        .drop(columns=OPTIMALITY_COLUMNS)\
        .drop(columns=AGREEMENT_COLUMNS)\
        .to_csv(results_csv_file.replace(".csv", "-mean-correct.csv"), index=True)
    results_mean\
        .drop(columns=OPTIMALITY_COLUMNS)\
        .drop(columns=CORRECTNESS_COLUMNS)\
        .to_csv(results_csv_file.replace(".csv", "-mean-agreement.csv"), index=True)

titleFontSize = 12
legendFontSize = 13
axesFontSize = 10
markerSize=12
style="g-o"

figsize=(12, 7)
dpi=80
facecolor='w'
edgecolor='k'

def plot_vs_std(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
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
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if num_of_voters==11:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig("{}/{}_vs_std.png".format(folder, column), format="png")
    plt.draw()


def plot_vs_mean(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
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
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if num_of_voters==11:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig("{}/{}_vs_mean.png".format(folder, column))
    plt.draw()


def plot_vs_voters(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    results = pandas.read_csv(results_csv_file)

    for index,expertise_mean in enumerate(expertise_means):
        results_for_mean = results.loc[results['mean']==expertise_mean]
        ax = plt.subplot(3, 3, index+1)
        ax.set_title('mean={}'.format(expertise_mean),
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_std in expertise_stds:
            results_for_std = results_for_mean.loc[results_for_mean['std']==expertise_std]
            x_values = results_for_std['voters']
            y_values = results_for_std[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="std={}".format(expertise_std))
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if expertise_mean==1:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig("{}/{}_vs_voters.png".format(folder, column))
    plt.draw()

