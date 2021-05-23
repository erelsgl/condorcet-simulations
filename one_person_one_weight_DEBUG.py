#!python3


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



def plot_vs_mean_on_one_page(results_csv_file:str, figure_file:str, 
    map_column_codes_to_column_names:dict,  num_of_voterss:list, expertise_means:list, expertise_stds:list, line_at_half:bool=False):
    A4 = (8,11) # A4 page: 8 inch length, 11 inch height
    plt.figure(figsize=A4, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)

    rows = len(num_of_voterss)
    cols = len(map_column_codes_to_column_names)+1
    top_left_axis = plt.subplot(rows, cols, 1)
    for row_index in [0,1]:
        subplot_index = row_index*cols+1
        ax = plt.subplot(rows, cols, subplot_index, sharex=top_left_axis, sharey=top_left_axis)
        ax.plot(range(100), range(100))

    # plt.show()
    plt.savefig(figure_file)


# num_of_voterss = [3, 5, 7, 9, 11, 21,]
num_of_voterss = [7,9]

expertise_means = [.55, .6, 0.65,
                   .7, .75,  .8,
                   .85, .9,  0.95]

expertise_stds = [0.08]


results_file = f"results/1000iters-beta.csv"

plot_vs_mean_on_one_page(results_file, "results/optimality_vs_mean.png",
    {
        "optimal_is_strong_epistocracy": "optimal_is_strong_epistocracy",
    },
    num_of_voterss, expertise_means, expertise_stds, line_at_half=True)
