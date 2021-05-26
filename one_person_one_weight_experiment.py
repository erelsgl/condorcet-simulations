#!python3

"""
A main program for a simulation experiment for the "one person, one weight" paper.
See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from experiment import *
import matplotlib.pyplot as plt


INDEX_COLUMNS = ["iterations","voters", "mean", "mean_bucket", "std", "std_bucket"]
OPTIMALITY_COLUMNS = [
                "optimal_is_strong_democracy", "optimal_is_weak_democracy",
                "optimal_is_weak_epistocracy", "optimal_is_strong_epistocracy",
                "optimal_is_expert_rule",     ]
AGREEMENT_COLUMNS = ["optimal_agrees_majority"]
REPORTED_COLUMNS = OPTIMALITY_COLUMNS + AGREEMENT_COLUMNS
TABLE_COLUMNS = INDEX_COLUMNS + REPORTED_COLUMNS


def create_results_revision(results_csv_file:str, num_of_iterations:int, num_of_voterss:list, expertise_means:list, expertise_stds:list, num_of_decisions:int=1, debug_committees=False,
    random_expertise_levels=beta_expertise_levels):
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
                print(f"{num_of_voters} voters, {expertise_mean} mean, {expertise_std} std", flush=True)
                optimal_strong_democracy_sum = 0
                optimal_weak_democracy_sum = 0
                optimal_weak_epistocracy_sum = 0
                optimal_strong_epistocracy_sum = 0
                optimal_expert_rule_sum = 0
                optimal_agrees_majority_sum = 0

                if num_of_voters >= 21 and num_of_iterations > 100:
                    num_of_iterations = 100
                for _ in range(num_of_iterations):
                    committee = Committee(random_expertise_levels(expertise_mean, expertise_std, num_of_voters))
                    if (debug_committees): print(committee)
                    if num_of_voters <= 21:
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

                results_table.add(OrderedDict((
                    ("iterations", num_of_iterations),
                    ("voters", num_of_voters),
                    ("mean", expertise_mean),
                    ("std", expertise_std),

                    ("optimal_is_strong_democracy", optimal_strong_democracy_sum / num_of_iterations),
                    ("optimal_is_weak_democracy",  optimal_weak_democracy_sum / num_of_iterations),
                    ("optimal_is_weak_epistocracy",  optimal_weak_epistocracy_sum / num_of_iterations),
                    ("optimal_is_strong_epistocracy",  optimal_strong_epistocracy_sum / num_of_iterations),
                    ("optimal_is_expert_rule", optimal_expert_rule_sum /num_of_iterations),
                    ("optimal_agrees_majority", optimal_agrees_majority_sum / num_of_iterations),
                )))
    results_table.done()






num_of_iterations = 1000

distribution="beta"
random_expertise_levels=beta_expertise_levels

# distribution="norm"
# random_expertise_levels=truncnorm_expertise_levels

num_of_voterss = [3, 5, 7, 9, 11, 21, 31, 41, 51]

expertise_means = [.55, .6, 0.65,
                   .7, .75,  .8,
                   .85, .9,  0.95]

expertise_stds = [0.02, 0.03, 0.04,
                  0.07, 0.08, 0.09,
                  0.12, 0.13, 0.14]

results_file = f"results/{num_of_iterations}iters-{distribution}.csv"
create_results_revision(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds,
    random_expertise_levels=random_expertise_levels
    )

