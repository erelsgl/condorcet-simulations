"""
A main program for running the experiments in the paper:

    "On the Degree of Generalizability of Condorcet Jury Theorem"

Since:  2020-06
Author: Erel Segal-Halevi

Credits: https://stats.stackexchange.com/q/471426/10760
"""

import pandas, logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

num_of_iterations = 1000
distribution="norm"


def add_discrete_derivative_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['mean', 'std', 'voters'])
    results['d_majority_correct_d_voters'] = results.groupby(['mean', 'std'])['majority_correct'].diff().fillna(0)
    results['d_optimal_correct_d_voters'] = results.groupby(['mean', 'std'])['optimal_correct'].diff().fillna(0)
    results.to_csv(results_csv_file, index=False)

def add_ratio_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['mean', 'std', 'voters'])
    results['ratio_difference_to_mean'] = results["majority_correct_minus_mean"] / results["optimal_correct_minus_mean"]
    results['ratio_derivative_by_n']    = results["d_majority_correct_d_voters"] / results["d_optimal_correct_d_voters"]
    results['ratio_correct']            = results["majority_correct"] / results["optimal_correct"]
    results.to_csv(results_csv_file, index=False)

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

    results_mean\
        .to_csv(results_csv_file.replace(".csv", "-groups.csv"), index=True)
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


def create_sample_results(results_csv_file:str, means:list, stds:list):
    results = pandas.read_csv(results_csv_file)

    # Create buckets for the mean:
    results = results.loc[results['mean'].isin(means)]
    results = results.loc[results['std'].isin(stds)]

    results\
        .to_csv(results_csv_file.replace(".csv", "-samples.csv"), index=False)
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



if __name__ == "__main__":
    results_csv_file = "degree_of_generalizability__results/" + f"{num_of_iterations}iters-{distribution}.csv"
    add_discrete_derivative_columns(results_csv_file)
    add_ratio_columns(results_csv_file)
    create_group_results(results_csv_file, 0.65, 0.8, 0.04, 0.09)
    create_sample_results(results_csv_file, means=[0.55, 0.75, 0.95], stds=[0.02, 0.08, 0.14])
