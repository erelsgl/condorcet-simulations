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

from collections import OrderedDict
import pandas, os.path

from Committee import Committee
from expertise_levels import truncnorm_expertise_levels, beta_expertise_levels

import logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)




def add_difference_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file)
    # results["majority_correct_minus_mean"] = results["majority_correct"] - results["mean"]
    # results["optimal_correct_minus_mean"] = results["optimal_correct"] - results["mean"]
    # results["majorityequal_correct_minus_mean"] = results["majority_correct_with_equal_expertise"] - results["mean"]
    results.to_csv(results_csv_file, index=True)

def add_discrete_derivative_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['mean', 'std', 'voters'])
    # results['d_majority_correct_d_voters'] = results.groupby(['mean', 'std'])['majority_correct'].diff().fillna(0)
    # results['d_optimal_correct_d_voters'] = results.groupby(['mean', 'std'])['optimal_correct'].diff().fillna(0)
    # results['d_majorityequal_correct_d_voters'] = results.groupby(['mean', 'std'])['majority_correct_with_equal_expertise'].diff().fillna(0)
    results.to_csv(results_csv_file, index=True)

def add_ratio_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['mean', 'std', 'voters'])
    results['ratio_difference_to_mean'] = results["majority_correct_minus_mean"] / results["optimal_correct_minus_mean"]
    results['ratio_derivative_by_n']    = results["d_majority_correct_d_voters"] / results["d_optimal_correct_d_voters"]
    results['ratio_correct']            = results["majority_correct"] / results["optimal_correct"]
    results.to_csv(results_csv_file, index=True)

def create_group_results(results_csv_file:str, mean_1:float, mean_2:float, std_1:float, std_2:float):
    results = pandas.read_csv(results_csv_file)

    # Create buckets for the mean:
    results["mean_bucket"] = "Lower"
    results.loc[results.query(f'mean > {mean_1}').index, "mean_bucket"] = "Medium"
    results.loc[results.query(f'mean > {mean_2}').index, "mean_bucket"] = "Upper"

    # Create buckets for the std:
    results["std_bucket"] = "Lower"
    results.loc[results.query(f'std > {std_1}').index, "std_bucket"] = "Medium"
    results.loc[results.query(f'std > {std_2}').index, "std_bucket"] = "Upper"

    results_mean = results.groupby(['voters', 'mean_bucket', 'std_bucket']).mean().round(3)
    results_mean.drop(columns=["iterations","mean","std"], inplace=True)
    results_mean.index.names = ["voters", "mean", "std"]

    # results_mean\
    #     .drop(columns=CORRECTNESS_COLUMNS)\
    #     .drop(columns=OLD_CORRECTNESS_COLUMNS)\
    #     .drop(columns=AGREEMENT_COLUMNS)\
    #     .rename(columns={
    #         "simple_majority_optimal": "smr",
    #         "minority_decisiveness_optimal": "mino-d",
    #         "non_tyrannic_minority_decisiveness_optimal": "n-t-m-d",
    #         "majority_tyranny_optimal": "majo-t",
    #         "minority_tyranny_optimal": "mino-t",
    #         "expert_tyranny_optimal": "expert",
    #         "minority_colluding": "min-coll"})\
    #     .to_csv(results_csv_file.replace(".csv", "-mean-optimal.csv"), index=True)
    # results_mean\
    #     .drop(columns=OPTIMALITY_COLUMNS)\
    #     .drop(columns=AGREEMENT_COLUMNS) \
    #     .drop(columns=OLD_CORRECTNESS_COLUMNS)\
    # .to_csv(results_csv_file.replace(".csv", "-mean-correct.csv"), index=True)
    # results_mean\
    #     .drop(columns=OPTIMALITY_COLUMNS)\
    #     .drop(columns=CORRECTNESS_COLUMNS)\
    #     .drop(columns=OLD_CORRECTNESS_COLUMNS)\
    #     .to_csv(results_csv_file.replace(".csv", "-mean-agreement.csv"), index=True)



if __name__=="__main__":
    # Make a small experiment, for debug
    num_of_iterations = 10
    num_of_voterss = [3,5,7,9,11]
    expertise_means = [0.55]
    expertise_stds = [0.03]
    results_file="results/debug.csv"
    create_results_revision(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)


