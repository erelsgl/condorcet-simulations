"""
A main program for running the experiments in the paper:

    "On the Degree of Generalizability of Condorcet Jury Theorem"
    by Roi Baharad, Shmuel Nitzan and Erel Segal-Halevi

Since:  2020-06
Author: Erel Segal-Halevi

Credits: https://stats.stackexchange.com/q/471426/10760

COLUMNS:

 * true_mean                                          == mu* 
 * majority_correct                                   == pi
 * d_majority_correct_d_voters                        == Delta_pi / Delta_n
 * majority_correct_minus_true_mean                   == pi - mu* 
 * optimal_correct                                    == pi* 
 * optimal_correct_minus_majority_correct             == pi* - pi
 * ratio_difference_to_mean == (pi - mu*)/(pi* - mu*) == gamma_M
 * ratio_correct                         == pi / pi*  == gamma_O
"""



import pandas, logging, sys, numpy as np
logger = logging.getLogger(__name__)

num_of_iterations = 1000


def add_discrete_derivative_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['distribution', 'mean', 'std', 'voters'])
    results['d_majority_correct_d_voters'] = np.round(results.groupby(['distribution', 'mean', 'std'])['majority_correct'].diff().fillna(0)/2, 3)
    results['d_optimal_correct_d_voters'] = np.round(results.groupby(['distribution', 'mean', 'std'])['optimal_correct'].diff().fillna(0)/2, 3)
    results.to_csv(results_csv_file, index=False)

def add_ratio_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['distribution', 'mean', 'std', 'voters'])
    results['ratio_difference_to_mean'] = np.round(results["majority_correct_minus_true_mean"] / results["optimal_correct_minus_true_mean"], 3)
    results['ratio_derivative_by_n']    = np.round(results["d_majority_correct_d_voters"] / results["d_optimal_correct_d_voters"], 3)
    results['ratio_correct']            = np.round(results["majority_correct"] / results["optimal_correct"], 3)
    results.to_csv(results_csv_file, index=False)

def add_uniform_distribution_column(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['distribution', 'mean', 'std', 'voters'])
    results.pop('uniform_distribution_from')
    results.insert(1, 'uniform_distribution_from', np.round(results["mean"] - results["std"]*np.sqrt(3),1))
    results.pop('uniform_distribution_to')
    results.insert(2, 'uniform_distribution_to',   np.round(results["mean"] + results["std"]*np.sqrt(3),1))
    results.to_csv(results_csv_file, index=False)

def round_columns(results_csv_file:str):
    results = pandas.read_csv(results_csv_file).sort_values(by=['distribution', 'mean', 'std', 'voters'])
    results['optimal_correct_minus_mean'] = np.round(results["optimal_correct_minus_mean"], 3)
    results['optimal_correct_minus_true_mean'] = np.round(results["optimal_correct_minus_true_mean"], 3)
    results['majority_correct_minus_mean'] = np.round(results["majority_correct_minus_mean"], 3)
    results['majority_correct_minus_true_mean'] = np.round(results["majority_correct_minus_true_mean"], 3)
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

    results_mean = results.groupby(['distribution', 'voters', 'mean_bucket', 'std_bucket']).mean().round(3)
    results_mean.drop(columns=["iterations","mean","std"], inplace=True)
    results_mean.index.names = ['distribution', "voters", "mean", "std"]

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

    results = results.round(3)
    results['mean'] = results['mean'].map({means[0]:"Lower", means[1]:"Medium", means[2]:"Upper"})
    results['std'] = results['std'].map({stds[0]:"Lower", stds[1]:"Medium", stds[2]:"Upper"})

    columns_1 = ['voters', 'mean', 'std', 'true_mean', 
        'majority_correct', 'd_majority_correct_d_voters', 'majority_correct_minus_true_mean',
        'optimal_correct', 'optimal_correct_minus_majority_correct',
        ]
    columns_2 = ['voters', 'mean', 'std', 
        'ratio_difference_to_mean', 'ratio_derivative_by_n', 'ratio_correct'
        ]

    results[columns_1].to_csv(results_csv_file.replace(".csv", "-table-1.csv"), index=False)
    results[columns_2].to_csv(results_csv_file.replace(".csv", "-table-2.csv"), index=False)



if __name__ == "__main__":
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    # Code for experiment in the original submission
    results_csv_file = "degree_of_generalizability__results/" + f"{num_of_iterations}iters-norm.csv"
    add_discrete_derivative_columns(results_csv_file)
    add_ratio_columns(results_csv_file)
    create_group_results(results_csv_file, 0.65, 0.8, 0.04, 0.09)
    create_sample_results(results_csv_file, means=[0.55, 0.75, 0.95], stds=[0.04, 0.08, 0.14])

    # Code for experiment in the original submission
    results_csv_file = "degree_of_generalizability__results/" + f"{num_of_iterations}iters.csv"
    add_discrete_derivative_columns(results_csv_file)
    add_ratio_columns(results_csv_file)
    add_uniform_distribution_column(results_csv_file)
    round_columns(results_csv_file)
