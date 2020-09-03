#!python3

"""
A main program for running a decision-rule experiment.
See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from experiment import *
import matplotlib.pyplot as plt


titleFontSize = 12
legendFontSize = 10
axesFontSize = 10
markerSize=12
style="g-o"

figsize=(16, 12)  # (length, height)
map_index_to_subplot_index = [1, 2, 3, 5, 6, 7, 9, 10, 11]
dpi=80
facecolor='w'
edgecolor='k'

rows = 3
cols = 4


def plot_vs_mean_on_one_page(results_csv_file:str, figure_file:str, columns: list, column_names: list, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=A4, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

    results = pandas.read_csv(results_csv_file)

    rows = len(num_of_voterss)
    cols = len(columns)+1
    top_left_axis = plt.subplot(rows, cols, 1)
    for row_index, num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results.loc[results['voters']==num_of_voters]
        for col_index, column in enumerate(columns):
            subplot_index = row_index*cols+col_index+1
            ax = plt.subplot(rows, cols, subplot_index, sharex=top_left_axis, sharey=top_left_axis)

            if row_index==0:
                ax.set_title(column_names[col_index],
                             fontsize=11, weight='bold')
            ax.set_xlabel('', fontsize=axesFontSize)

            for expertise_std in expertise_stds:
                results_for_std = results_for_voters.loc[results_for_voters['std']==expertise_std]
                x_values = results_for_std['mean']
                y_values = results_for_std[column]
                ax.plot(x_values, y_values, markersize=markerSize, label="std={}".format(expertise_std))

                if line_at_half:
                    ax.plot(x_values, x_values, color="black", label="")  # show the mean
                    # ax.plot(x_values, [0.5]*len(x_values), color="black", label="")

            if row_index==2 and col_index==cols-2:
                ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(1.5, 1))

            if col_index==cols-2:
                ax.text(x=max(expertise_means)*1.1, y=max(y_values)/2, s='n={}'.format(num_of_voters))

            if col_index>0:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row_index<rows-1:
                plt.setp(ax.get_xticklabels(), visible=False)

    plt.xlabel("mean")
    folder, _ = os.path.split(results_csv_file)
    plt.savefig(figure_file)
    plt.draw()



def plot_vs_voters_on_one_page(results_csv_file:str, figure_file:str, columns: list, column_names: list, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=A4, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

    results = pandas.read_csv(results_csv_file)

    rows = len(expertise_means)
    cols = len(columns)+1
    top_left_axis = plt.subplot(rows, cols, 1)
    for row_index, expertise_mean in enumerate(expertise_means):
        results_for_mean = results.loc[results['mean']==expertise_mean]
        for col_index, column in enumerate(columns):
            subplot_index = row_index*cols+col_index+1
            ax = plt.subplot(rows, cols, subplot_index, sharex=top_left_axis, sharey=top_left_axis)

            if row_index==0:
                ax.set_title(column_names[col_index],
                             fontsize=11, weight='bold')
            ax.set_xlabel('', fontsize=axesFontSize)

            for expertise_std in expertise_stds:
                results_for_std = results_for_mean.loc[results_for_mean['std']==expertise_std]
                x_values = results_for_std['voters']
                y_values = results_for_std[column]
                ax.plot(x_values, y_values, markersize=markerSize, label="std={}".format(expertise_std))

                if max(y_values)>0.5 or line_at_half:
                    ax.plot(x_values, [0.5]*len(x_values), color="black", label="")

            if row_index==2 and col_index==cols-2:
                ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(1.5, 1))

            if col_index==cols-2:
                ax.text(x=max(num_of_voterss)*1.1, y=max(y_values)/2, s='mean={}'.format(expertise_mean))

            if col_index>0:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row_index<rows-1:
                plt.setp(ax.get_xticklabels(), visible=False)

    plt.xlabel("#voters")
    folder, _ = os.path.split(results_csv_file)
    plt.savefig(figure_file)
    plt.draw()



num_of_iterations = 1000

num_of_voterss = [3, 5, 7,
                  9, 11, 21,
                  31, 41, 51]
results_file="results/{}iters.csv".format(num_of_iterations)

expertise_means = [.5, .55, .6,
                   .7, .75, .8,
                   .9, .95, 1]

expertise_stds = [0.02, 0.03, 0.04,
                  0.06, 0.07, 0.08,
                  0.10, 0.11, 0.12]


# create_results(results_file, num_of_iterations, num_of_voterss, expertise_means, expertise_stds)
add_difference_columns(results_file)
add_discrete_derivative_columns(results_file)
add_ratio_columns(results_file)
create_group_results(results_file)
exit(0)

plot_vs_mean_on_one_page(results_file, "results/correctness_vs_mean.png",
     ["majority_correct", "optimal_correct"],
     ["SMR correct", "Optimal correct"],
     [3,5,7,9,11,21], expertise_means, expertise_stds, line_at_half=True)

plot_vs_mean_on_one_page(results_file, "results/correctness_vs_mean_1.png",
     ["optimal_correct_minus_majority_correct"],
     ["Optimal minus SMR"],
     [3,5,7,9,11,21], expertise_means, expertise_stds, line_at_half=False)

plot_vs_voters_on_one_page(results_file, "results/correctness_vs_voters.png",
     ["majority_correct", "optimal_correct"],
     ["SMR correct", "Optimal correct"],
     num_of_voterss, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], expertise_stds, line_at_half=True)

plot_vs_voters_on_one_page(results_file, "results/correctness_vs_voters_1.png",
     ["optimal_correct_minus_majority_correct", "d_majority_correct_d_voters"],
     ["Optimal minus SMR", "SMR derivative"],
     num_of_voterss, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], expertise_stds, line_at_half=False)

plt.close()
