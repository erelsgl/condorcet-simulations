""" 
A main program for running the experiments in the paper:

"One person, one weight: when is weighted voting democratic?" (SCWE, 2022)

See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from expertise_levels import truncnorm_expertise_levels, beta_expertise_levels
from Committee import Committee


num_of_iterations = 1000

distribution="beta"
random_expertise_levels=beta_expertise_levels

# distribution="norm"
# random_expertise_levels=truncnorm_expertise_levels


def create_results_revision(voters:int, mean:float, std:float, num_of_decisions:int=1, debug_committees=False):
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
    optimal_strong_democracy_sum = 0
    optimal_weak_democracy_sum = 0
    optimal_weak_epistocracy_sum = 0
    optimal_strong_epistocracy_sum = 0
    optimal_expert_rule_sum = 0
    optimal_agrees_majority_sum = 0

    if voters >= 21 and num_of_iterations > 100:
        num_of_iterations = 100
    for _ in range(num_of_iterations):
        committee = Committee(random_expertise_levels(mean, std, voters))
        if (debug_committees): print(committee)
        if voters <= 21:
            is_optimal_strong_democracy = committee.is_optimal_strong_democracy()
            is_optimal_weak_democracy = committee.is_optimal_weak_democracy()
            is_optimal_weak_epistocracy = not is_optimal_weak_democracy
            is_optimal_strong_epistocracy = committee.is_optimal_strong_epistocracy()

            if is_optimal_strong_democracy and is_optimal_strong_epistocracy:
                raise ValueError(f"Both sd and se are optimal - bug! committee={committee}")

            optimal_strong_democracy_sum += is_optimal_strong_democracy
            optimal_weak_democracy_sum   += (is_optimal_weak_democracy  and not is_optimal_strong_democracy)
            optimal_weak_epistocracy_sum += (is_optimal_weak_epistocracy and not is_optimal_strong_epistocracy)
            optimal_strong_epistocracy_sum += is_optimal_strong_epistocracy
            optimal_expert_rule_sum += committee.is_optimal_minority_decisiveness(minority_size=1)

        for _ in range(num_of_decisions):
            vote = committee.vote()
            optimal_vote  = committee.optimal_weighted_rule(vote)
            majority_vote = committee.simple_majority_rule(vote)
            optimal_agrees_majority_sum += (optimal_vote==majority_vote)/num_of_decisions

    return {
        "iterations": num_of_iterations,
        "optimal_is_strong_democracy": optimal_strong_democracy_sum / iterations,
        "optimal_is_weak_democracy":  optimal_weak_democracy_sum / iterations,
        "optimal_is_weak_epistocracy":  optimal_weak_epistocracy_sum / iterations,
        "optimal_is_strong_epistocracy":  optimal_strong_epistocracy_sum / iterations,
        "optimal_is_expert_rule": optimal_expert_rule_sum /iterations,
        "optimal_agrees_majority": optimal_agrees_majority_sum / iterations,
    }




if __name__ == "__main__":
    import logging, experiments_csv

    experiment = experiments_csv.Experiment("results/", f"{num_of_iterations}iters-{distribution}.csv", "results/backups/")
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
    experiment.run(create_results_revision, input_ranges)
