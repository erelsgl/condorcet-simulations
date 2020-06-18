#!python3

"""
A main program for running a decision-rule experiment.
See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from experiment import *


titleFontSize = 12
legendFontSize = 13
axesFontSize = 10
markerSize=12
style="g-o"

figsize=(16, 12)  # (length, height)
map_index_to_subplot_index = [1, 2, 3, 5, 6, 7, 9, 10, 11]
dpi=80
facecolor='w'
edgecolor='k'


def plot_vs_std(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    results = pandas.read_csv(results_csv_file)

    for index,num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results.loc[results['voters']==num_of_voters]
        ax = plt.subplot(3, 4, map_index_to_subplot_index[index])
        ax.set_title('{} voters'.format(num_of_voters),
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_mean in expertise_means:
            results_for_mean = results_for_voters.loc[results_for_voters['mean']==expertise_mean]
            x_values = results_for_mean['std']
            y_values = results_for_mean[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="mean={}".format(expertise_mean))
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if index==5:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig("{}/{}_vs_std.png".format(folder, column), format="png")
    plt.draw()


def plot_vs_mean(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    results = pandas.read_csv(results_csv_file)

    for index,num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results.loc[results['voters']==num_of_voters]
        ax = plt.subplot(3, 4, map_index_to_subplot_index[index])
        ax.set_title('{} voters'.format(num_of_voters),
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_std in expertise_stds:
            results_for_std = results_for_voters.loc[results_for_voters['std']==expertise_std]
            x_values = results_for_std['mean']
            y_values = results_for_std[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="std={}".format(expertise_std))
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if index==5:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig("{}/{}_vs_mean.png".format(folder, column))
    plt.draw()


def plot_vs_voters(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    results = pandas.read_csv(results_csv_file)

    for index,expertise_mean in enumerate(expertise_means):
        results_for_mean = results.loc[results['mean']==expertise_mean]
        ax = plt.subplot(3, 4, map_index_to_subplot_index[index])
        ax.set_title('mean={}'.format(expertise_mean),
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_std in expertise_stds:
            results_for_std = results_for_mean.loc[results_for_mean['std']==expertise_std]
            x_values = results_for_std['voters']
            y_values = results_for_std[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="std={}".format(expertise_std))
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if index==5:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig("{}/{}_vs_voters.png".format(folder, column))
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
create_group_results(results_file)
for column in REPORTED_COLUMNS:
    plot_vs_mean(results_file, column, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    plot_vs_std(results_file, column, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    plot_vs_voters(results_file, column, num_of_voterss, expertise_means, expertise_stds, line_at_half=False)
    plt.close()
