"""
A main program for running the experiments in the paper:

    "On the Degree of Generalizability of Condorcet Jury Theorem"
    by Roi Baharad, Shmuel Nitzan and Erel Segal-Halevi

Since:  2020-04
Author: Erel Segal-Halevi

Credits: https://stats.stackexchange.com/q/471426/10760
"""

from Committee import Committee
import expertise_levels

import logging, sys, numpy as np
logger = logging.getLogger(__name__)


num_of_iterations = 1000

def create_results(voters:int, mean:float, std:float, distribution:callable, num_of_decisions:int=1):
    """
    Run an experiment with voters of different expertise level.

    There are `num_of_voters` voters drawn at random.
    The expertise of a voter is the probability that the voter is correct.
    It is a truncated-normal random variable, with mean  `expertise_mean`  and std  `expertise_std,
       truncated to be in the range [0.5 , 1].

    :param voters:      the number of voters in the committee. Should be an odd integer.
    :param mean:        the mean expertise-level  (should be between 0.5 and 1).
    :param std:         the std of the expertise level.
    :param num_of_decisions:  how many times to run the experiment. Default=1.
    :param debug_committees

    NOTE: num_decisions is set to 1 based on the answer by Henry:
            https://stats.stackexchange.com/a/471431/10760
    """
    optimal_correct_sum = 0
    majority_correct_sum = 0

    if hasattr(distribution, "true_mean"):
        true_mean = distribution.true_mean(mean,std)
    else:
        true_mean = mean
    if hasattr(distribution, "true_std"):
        true_std = distribution.true_std(mean,std)
    else:
        true_std = std

    for _ in range(num_of_iterations):
        committee = Committee(distribution(mean, std, voters))
        logger.debug("Committee: %s", committee)
        for _ in range(num_of_decisions):
            vote = committee.vote()
            optimal_vote  = committee.optimal_weighted_rule(vote)
            majority_vote = committee.simple_majority_rule(vote)
            optimal_correct_sum  += optimal_vote/num_of_decisions
            majority_correct_sum += majority_vote/num_of_decisions

    return {
        "iterations": num_of_iterations,
        "voters": voters,
        "mean": mean,
        "std": std,
        "true_mean": true_mean,
        "true_std": true_std,

        "optimal_correct": optimal_correct_sum / num_of_iterations,                      # pi*
        "optimal_correct_minus_mean": optimal_correct_sum / num_of_iterations - mean,
        "optimal_correct_minus_true_mean": optimal_correct_sum / num_of_iterations - true_mean,
        "majority_correct": majority_correct_sum / num_of_iterations,                    # pi
        "majority_correct_minus_mean": majority_correct_sum / num_of_iterations - mean,
        "majority_correct_minus_true_mean": majority_correct_sum / num_of_iterations - true_mean,
        "optimal_correct_minus_majority_correct": (optimal_correct_sum-majority_correct_sum) / num_of_iterations,
    }


def run_experiment_truncnorm(folder, backup_folder):
    """
    The experiment from the original submission
    """
    filename = f"{num_of_iterations}iters-truncnorm.csv"
    experiment = experiments_csv.Experiment(folder, filename, backup_folder)
    experiment.logger.setLevel(logging.INFO)

    input_ranges_original_submission = {
        "voters": [3, 5, 7, 9, 11, 21, 31, 41, 51],
        "mean": [.55, .6, 0.65,
                .7, .75,  .8,
                .85, .9,  0.95],
        "std": [0.02, 0.04,
                0.08, 0.16, 0.32,
                0.64, 1.28, 2.56],
        "distribution": [
            expertise_levels.TruncNorm(0.501,0.999),
            expertise_levels.TruncNorm(0.001,0.999),
            ],
    }
    experiment.run(create_results, input_ranges_original_submission)

def run_experiment_uniform(folder, backup_folder):
    """
    An experiment for the first revision: Uniform distribution
    """
    filename = f"{num_of_iterations}iters-uniform.csv"
    experiment = experiments_csv.Experiment(folder, filename, backup_folder)
    experiment.logger.setLevel(logging.INFO)

    interval_starts = [0.51, 0.6, 0.7, 0.8, 0.9]
    interval_ends   = [0.6, 0.7, 0.8, 0.9, 0.99]
    for start in interval_starts:
        for end in interval_ends:
            if end>start:
                print(f"Interval [{start}, {end}]")
                mean = np.round((start+end)/2,2)
                std = np.round((end-start)/np.sqrt(12),3)
                input_ranges = {
                    "voters": [3, 5, 7, 9, 11, 21, 31, 41, 51],
                    "distribution": [expertise_levels.uniform],
                    "mean": [mean],
                    "std":  [std],
                }
                experiment.run(create_results, input_ranges)
    
def run_experiment_beta(folder, backup_folder):
    """
    Experiment for first revision: Beta distribution
    """
    filename = f"{num_of_iterations}iters-beta.csv"
    experiment = experiments_csv.Experiment(folder, filename, backup_folder)
    experiment.logger.setLevel(logging.INFO)

    input_ranges = {
        "voters": [3, 5, 7, 9, 11, 21, 31, 41, 51],
        "distribution": [expertise_levels.beta],
        "mean": [8/14, 9/14, 10/14],
        "std":  [1.1/14],
    }
    experiment.run(create_results, input_ranges)


if __name__ == "__main__":
    import logging, experiments_csv

    np.seterr(all="raise")
    np.seterr(under='ignore')       # Ignore underflow errors

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)  # to log the committeesx

    folder = "degree_of_generalizability__results/"
    backup_folder = f"{folder}backups/"

    # run_experiment_truncnorm(folder, backup_folder)
    run_experiment_uniform(folder, backup_folder)
    run_experiment_beta(folder, backup_folder)
