#!python3

"""
A main program for debugging a decision-rule experiment.
See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from experiment import *

num_of_iterations = 10
num_of_voterss = [61]
expertise_means = [0.55]
expertise_stds = [0.03]
results_file="results/debug.csv"
create_results(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)
