#!python3

"""
A main program for a simulation experiment for the "one person, one weight" paper.
See experiment.py for details.

Since:  2020-06
Author: Erel Segal-Halevi
"""

from experiment import *
import numpy as np
import matplotlib.pyplot as plt


titleFontSize = 12
legendFontSize = 10
axesFontSize = 10
markerSize=12
style="g-o"

figsize=(16, 12)  # (length, height)
map_index_to_subplot_index = [1, 2, 3, 5, 6, 7, 9, 10, 11]
dpi=300
facecolor='w'
edgecolor='k'

rows = 3
cols = 4


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
        .rename(columns={
            "optimal_is_strong_democracy": "strong\ndemocracy",
            "optimal_is_weak_democracy": "weak\ndemocracy",
            "optimal_is_weak_epistocracy": "weak\nepistocracy",
            "optimal_is_strong_epistocracy": "strong\nepistocracy",
            "optimal_is_expert_rule": "expert\nrule",
            "optimal_agrees_majority": "optimal\n=majority"})\
        .to_csv(results_csv_file.replace(".csv", "-mean-optimal.csv"), index=True)


map_column_codes_to_column_names = {
        "optimal_is_strong_democracy": "strong demo.",
        "optimal_is_weak_democracy": "weak demo.",
        "optimal_is_weak_epistocracy": "weak epis.",
        "optimal_is_strong_epistocracy": "strong epis.",
    }

map_column_codes_to_short_names = {
        "optimal_is_strong_democracy": "d",
        "optimal_is_weak_democracy": "sd",
        "optimal_is_weak_epistocracy": "se",
        "optimal_is_strong_epistocracy": "e",
    }

map_column_codes_to_colors = {
        "optimal_is_strong_democracy": (.99,.99,.8),
        "optimal_is_weak_democracy": (.7,.9,.7),
        "optimal_is_weak_epistocracy": (.6,.6,.8),
        "optimal_is_strong_epistocracy": (.5,.3,.3),
    }


def plot_vs_mean_on_one_page(results_csv_file:str, figure_file:str, 
    map_column_codes_to_column_names:dict,  num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=A4, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

    results = pandas.read_csv(results_csv_file)

    rows = len(num_of_voterss)
    cols = len(map_column_codes_to_column_names)+1
    # top_left_axis = plt.subplot(rows, cols, 1)
    for row_index, num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results.loc[results['voters']==num_of_voters]
        for col_index, (column,column_name) in enumerate(map_column_codes_to_column_names.items()):
            subplot_index = row_index*cols+col_index+1
            # using sharex and sharey causes a strange bug: "AttributeError: 'NoneType' object has no attribute 'canvas'"
            # ax = plt.subplot(rows, cols, subplot_index, sharex=top_left_axis, sharey=top_left_axis)
            ax = plt.subplot(rows, cols, subplot_index)
            ax.set_ylim((0,1))

            if row_index==0:
                ax.set_title(column_name,
                             fontsize=11, weight='bold')
            ax.set_xlabel('', fontsize=axesFontSize)

            for expertise_std in expertise_stds:
                results_for_std = results_for_voters.loc[results_for_voters['std']==expertise_std]
                x_values = results_for_std['mean']
                y_values = results_for_std[column]
                ax.plot(x_values, y_values, markersize=markerSize, label=f"std={expertise_std}")

                if line_at_half:
                    ax.plot(x_values, x_values, color="black", label="")  # show the mean
                    # ax.plot(x_values, [0.5]*len(x_values), color="black", label="")

            if row_index==2 and col_index==cols-2:
                ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(1.5, 1))

            if col_index==cols-2:
                ax.text(x=max(expertise_means)*1.1, y=max(y_values)/2, s=f'n={num_of_voters}')

            if col_index>0:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row_index<rows-1:
                plt.setp(ax.get_xticklabels(), visible=False)

    plt.xlabel("mean")
    # plt.show()
    plt.savefig(figure_file)

def plot_vs_voters_on_one_page(results_csv_file:str, figure_file:str, 
    map_column_codes_to_column_names:dict, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=A4, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

    results = pandas.read_csv(results_csv_file)

    rows = len(expertise_means)
    cols = len(map_column_codes_to_column_names)+1
    # top_left_axis = plt.subplot(rows, cols, 1)
    for row_index, expertise_mean in enumerate(expertise_means):
        results_for_mean = results.loc[results['mean']==expertise_mean]
        for col_index, (column,column_name) in enumerate(map_column_codes_to_column_names.items()):
            subplot_index = row_index*cols+col_index+1
            # ax = plt.subplot(rows, cols, subplot_index, sharex=top_left_axis, sharey=top_left_axis)
            ax = plt.subplot(rows, cols, subplot_index)

            if row_index==0:
                ax.set_title(column_name,
                             fontsize=11, weight='bold')
            ax.set_xlabel('', fontsize=axesFontSize)
            ax.set_ylim((0,1))

            for expertise_std in expertise_stds:
                results_for_std = results_for_mean.loc[results_for_mean['std']==expertise_std]
                x_values = results_for_std['voters']
                y_values = results_for_std[column]
                ax.plot(x_values, y_values, markersize=markerSize, label=f"std={expertise_std}")

                if max(y_values)>0.5 or line_at_half:
                    ax.plot(x_values, [0.5]*len(x_values), color="black", label="")

            if row_index==2 and col_index==cols-2:
                ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(1.5, 1))

            if col_index==cols-2:
                ax.text(x=max(num_of_voterss)*1.1, y=max(y_values)/2, s=f'mean={expertise_mean}')

            if col_index>0:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row_index<rows-1:
                plt.setp(ax.get_xticklabels(), visible=False)

    plt.xlabel("#voters")
    folder, _ = os.path.split(results_csv_file)
    plt.savefig(figure_file)
    plt.draw()



def pie_by_columns(rows:int, cols:int, row_index:int, col_index:int, row_title:str, col_title:str, values:pandas.DataFrame):
    subplot_index = row_index*cols+col_index+1
    fractions = []
    labels = []
    colors = []
    for column,column_name in map_column_codes_to_short_names.items():
        fraction = values[column].mean()
        fractions.append(fraction)
        labels.append(column_name if fraction>0.05 else ""  )
        colors.append(map_column_codes_to_colors[column])
    ax = plt.subplot(rows, cols, subplot_index)
    ax.pie(fractions, labels=labels, labeldistance=0.3, autopct="", normalize=True, radius=1.3, colors=colors,
        textprops={'fontsize': 8})
    if row_index==0:
        ax.set_title(col_title, fontsize=8, fontweight='normal')
    if col_index==cols-2:
        ax.text(x=1.5, y=0, s=row_title)
    if np.abs(1-sum(fractions))>0.05:
        ax.set_title(f"fractions={fractions}, sum={sum(fractions)}")
    return ax


def pie_by_voters_and_stds(results_csv_file:str, figure_file:str, map_column_codes_to_column_names:dict,  
    num_of_voterss:list, expertise_means:list, expertise_stds:list):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=(8,8), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    plt.title("Rules by #voters (n) and expertise standard deviation (s)")
    results = pandas.read_csv(results_csv_file)
    results_for_means = results[results["mean"].isin(expertise_means)]
    rows = len(num_of_voterss)
    cols = len(expertise_stds)+1
    for row_index, num_of_voters in enumerate(num_of_voterss):
        results_for_voters = results_for_means.loc[results_for_means['voters']==num_of_voters]
        for col_index, expertise_std in enumerate(expertise_stds):
            results_for_std = results_for_voters.loc[results_for_voters['std']==expertise_std]
            pie_by_columns(rows, cols, row_index, col_index, f"n={num_of_voters}", f"s={expertise_std}", results_for_std)
    plt.savefig(figure_file)



def pie_by_means_and_stds(results_csv_file:str, figure_file:str, 
    num_of_voterss:list, expertise_means:list, expertise_stds:list):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=(8,8), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    results = pandas.read_csv(results_csv_file)
    results_for_voters = results[results["voters"].isin(num_of_voterss)]
    rows = len(expertise_means)
    cols = len(expertise_stds)+1
    # plt.suptitle(f"Rules by expertise mean (m) and standard deviation (s), n={num_of_voterss}")
    plt.suptitle(f"{num_of_voterss[0]} voters")
    for row_index, expertise_mean in enumerate(expertise_means):
        results_for_mean = results_for_voters.loc[results_for_voters['mean']==expertise_mean]
        for col_index, expertise_std in enumerate(expertise_stds):
            results_for_std = results_for_mean.loc[results_for_mean['std']==expertise_std]
            pie_by_columns(rows, cols, row_index, col_index, f"m={expertise_mean}", f"s={expertise_std}", results_for_std)
    plt.savefig(figure_file)



def plot_vs_voters(results_csv_file:str, column: str, num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    results = pandas.read_csv(results_csv_file)

    for index,expertise_mean in enumerate(expertise_means):
        results_for_mean = results.loc[results['mean']==expertise_mean]
        ax = plt.subplot(rows, cols, map_index_to_subplot_index[index])
        ax.set_title(f'mean={expertise_mean}',
                     fontsize=titleFontSize, weight='bold')
        ax.set_xlabel('', fontsize=axesFontSize)

        for expertise_std in expertise_stds:
            results_for_std = results_for_mean.loc[results_for_mean['std']==expertise_std]
            x_values = results_for_std['voters']
            y_values = results_for_std[column]
            ax.plot(x_values, y_values, markersize=markerSize, label="std={expertise_std}")
            if max(y_values)>0.5 or line_at_half:
                ax.plot(x_values, [0.5]*len(x_values), color="black", label="")
        if index==5:
            ax.legend(prop={'size': legendFontSize}, bbox_to_anchor=(2, 1))

    plt.xticks(x_values.tolist(), fontsize=axesFontSize)
    folder, _ = os.path.split(results_csv_file)
    plt.savefig(f"{folder}/{column}_vs_voters.png")
    plt.draw()


num_of_voterss = [3, 5, 7, 9, 11, 21]

expertise_means = [.55, .6, 0.65,
                   .7, .75,  .8,
                   .85, .9,  0.95]

expertise_stds = [0.02, 0.03, 0.04,
                  0.07, 0.08, 0.09,
                  0.12, 0.13, 0.14]


# create_group_results(f"results/1000iters-norm.csv", mean_1=0.67, mean_2=0.82, std_1=0.05, std_2=0.1)
# create_group_results(f"results/1000iters-beta.csv", mean_1=0.67, mean_2=0.82, std_1=0.05, std_2=0.1)

# distribution="beta"
distribution="norm"
results_file = f"results/1000iters-{distribution}.csv"

for num_of_voters in num_of_voterss:
    print(f"{num_of_voters} voters", flush=True)
    pie_by_means_and_stds(results_file, f"results/optimality_by_means_and_stds_{num_of_voters}-{distribution}.png",
        num_of_voterss=[num_of_voters], 
        expertise_means=expertise_means,#[.55,.65,.75,.85,.95],
        expertise_stds=expertise_stds,#[0.02, 0.04, 0.08, 0.14]
        )
