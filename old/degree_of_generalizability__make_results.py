"""
A utility for performing simulation experiments on decision rules.

There are voters with different expertise level,
and it is required to decide whether the majority is correct,
or - alternatively - whether it is good to give the minority a decisive power.

Since:  2020-04
Author: Erel Segal-Halevi

Credits: https://stats.stackexchange.com/q/471426/10760
"""

from Committee import Committee
from expertise_levels import truncnorm_expertise_levels, beta_expertise_levels

import pandas, logging, sys
logger = logging.getLogger(__name__)

logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


num_of_iterations = 1000

# distribution="beta"
# random_expertise_levels=beta_expertise_levels

distribution="norm"
random_expertise_levels=truncnorm_expertise_levels


def create_results(voters:int, mean:float, std:float, num_of_decisions:int=1, debug_committees=False):
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
    # INDEX_COLUMNS = ["iterations","voters", "mean", "mean_bucket", "std", "std_bucket"]

    # OPTIMALITY_COLUMNS = [
    #                 "simple_majority_optimal", "minority_decisiveness_optimal",
    #                 "non_tyrannic_minority_decisiveness_optimal",
    #                 "majority_tyranny_optimal", "minority_tyranny_optimal", "expert_tyranny_optimal", "minority_colluding"]

    # CORRECTNESS_COLUMNS = ["optimal_correct", "majority_correct",
    #                     "optimal_correct_minus_mean", "majority_correct_minus_mean",
    #                     "d_optimal_correct_d_voters", "d_majority_correct_d_voters",
    #                     "ratio_difference_to_mean", "ratio_derivative_by_n", "ratio_correct",
    #                     "majority_correct_with_equal_expertise"]


    # AGREEMENT_COLUMNS = ["optimal_agrees_majority", "compromise_minority_agrees_majority", "compromise_minority_agrees_optimal", "compromose_strongmajority_agrees_majority", "compromose_strongmajority_agrees_optimal"]

    # REPORTED_COLUMNS = OPTIMALITY_COLUMNS + CORRECTNESS_COLUMNS + AGREEMENT_COLUMNS

    # TABLE_COLUMNS = INDEX_COLUMNS + REPORTED_COLUMNS


    minority_decisiveness_optimal = 0
    majority_tyranny_optimal = 0
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
        committee = Committee(truncnorm_expertise_levels(mean, std, voters))
        if (debug_committees): print(committee)
        minority_decisiveness_optimal += committee.is_optimal_minority_decisiveness()
        if (voters<=21):
            majority_tyranny_optimal += committee.is_optimal_majority_tyranny()
        if (voters<=51):
            minority_tyranny_optimal += committee.is_optimal_strong_epistocracy()
        expert_tyranny_optimal_sum += committee.is_optimal_minority_decisiveness(minority_size=1)
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

    return { 
        "iterations": num_of_iterations,
        "voters": voters,
        "mean": mean,
        "std": std,

        #  majority rule is optimal iff it is NOT optimal to let colluding minority decide:
        "simple_majority_optimal": (num_of_iterations-minority_decisiveness_optimal)/num_of_iterations,
        "minority_decisiveness_optimal":  minority_decisiveness_optimal / num_of_iterations,

        #  minority decisiveness is optimal iff colluding minority is optimal but a meritocratic rule is NOT optimal:
        "non_tyrannic_minority_decisiveness_optimal":  (minority_decisiveness_optimal - minority_tyranny_optimal) / num_of_iterations,
        "majority_tyranny_optimal": majority_tyranny_optimal/num_of_iterations,
        "minority_tyranny_optimal": minority_tyranny_optimal/num_of_iterations,
        "expert_tyranny_optimal": expert_tyranny_optimal_sum/num_of_iterations,

        "minority_colluding": minority_colluding_sum/num_of_iterations,

        "optimal_correct": optimal_correct_sum / num_of_iterations,
        "optimal_correct_minus_mean": optimal_correct_sum / num_of_iterations - mean,
        "majority_correct": majority_correct_sum / num_of_iterations,
        "majority_correct_minus_mean": majority_correct_sum / num_of_iterations - mean,
        "optimal_correct_minus_majority_correct": (optimal_correct_sum-majority_correct_sum) / num_of_iterations,
        "expert_correct": expert_correct_sum / num_of_iterations,
        "compromise_minority_correct": compromise_minority_correct_sum / num_of_iterations,
        "compromose_strongmajority_correct": compromose_strongmajority_correct_sum / num_of_iterations,

        "optimal_agrees_majority": optimal_agrees_majority_sum / num_of_iterations,
        "compromise_minority_agrees_majority": compromise_minority_agrees_majority_sum / num_of_iterations,
        "compromise_minority_agrees_optimal": compromise_minority_agrees_optimal_sum / num_of_iterations,
        "compromose_strongmajority_agrees_majority": compromose_strongmajority_agrees_majority_sum / num_of_iterations,
        "compromose_strongmajority_agrees_optimal": compromose_strongmajority_agrees_optimal_sum / num_of_iterations,
    }



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



# if __name__=="__main__":
#     # Make a small experiment, for debug
#     num_of_iterations = 10
#     num_of_voterss = [3,5,7,9,11]
#     expertise_means = [0.55]
#     expertise_stds = [0.03]
#     results_file="results/debug.csv"
#     create_results_revision(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)



if __name__ == "__main__":
    import logging, experiments_csv

    experiment = experiments_csv.Experiment("degree_of_generalizability__results/", f"{num_of_iterations}iters-{distribution}.csv", "degree_of_generalizability__results/backups/")
    experiment.logger.setLevel(logging.INFO)
    input_ranges = {
        "voters": [3, 5, 7, 9, 11, 21, 31, 41, 51],
        "mean": [.55, .6, 0.65,
                .7, .75,  .8,
                .85, .9,  0.95],
        "std": [0.02, 0.03, 0.04,
                0.07, 0.08, 0.09,
                0.12, 0.13, 0.14],
    }
    experiment.run(create_results, input_ranges)
