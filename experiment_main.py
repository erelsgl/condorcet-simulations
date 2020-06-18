#!python3

"""
A main program for running a decision-rule experiment.
See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from experiment import *

num_of_iterations = 1000

num_of_voterss = [3, 5, 7, 9, 11]
results_file="results/{}iters.csv".format(num_of_iterations)

# num_of_voterss = [3, 7, 15, 31, 63]
# results_file="results-manyvoters/{}iters.csv".format(num_of_iterations)

expertise_means = [.5, .55, .6,
                   .7, .75, .8,
                   .9, .95, 1]

expertise_stds = [0.02, 0.03, 0.04,
                  0.06, 0.07, 0.08,
                  0.10, 0.11, 0.12]



# create_results(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)
create_group_results(results_file)
for column in REPORTED_COLUMNS:
    plot_vs_mean(results_file, column, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    plot_vs_std(results_file, column, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    plot_vs_voters(results_file, column, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    plt.close()
