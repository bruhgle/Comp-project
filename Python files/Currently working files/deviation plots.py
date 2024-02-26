import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

num_days = int(1.627312e8 / 86400)

# Load deviation lists from the saved files
deviation_lists500 = np.load('deviation500.npy')
deviation_lists1000 = np.load('deviation1000.npy')
deviation_lists2000 = np.load('deviation2000.npy')
deviation_lists3000 = np.load('deviation3000.npy')
deviation_lists4000 = np.load('deviation4000.npy')

font_size = 16
size = (7, 5.5)

legend_elements1 = [
        Line2D([0], [0], linewidth=2, linestyle='-', color='darkred', markersize=20, label='Unaltered path'),
        Line2D([0], [0], linewidth=2, linestyle='-', color='red', markersize=20, label='15kms '),
        Line2D([0], [0], linewidth=2, linestyle='-', color='darkorange', markersize=20, label='30kms '),
        Line2D([0], [0], linewidth=2, linestyle='-', color='yellow', markersize=20, label='45kms '),
        Line2D([0], [0], linewidth=2, linestyle='-', color='limegreen', markersize=20, label='60kms '),
        Line2D([0], [0], linewidth=2, linestyle='--', color='red', markersize=20, label='Close approach'),
    ]

legend_elements2 = [
    Line2D([0], [0], linewidth=2, linestyle='-', color='fuchsia', markersize=20, label='500s'),
    Line2D([0], [0], linewidth=2, linestyle='-', color='blueviolet', markersize=20, label='1000s'),
    Line2D([0], [0], linewidth=2, linestyle='-', color='blue', markersize=20, label='2000s'),
    Line2D([0], [0], linewidth=2, linestyle='-', color='dodgerblue', markersize=20, label='3000s'),
    Line2D([0], [0], linewidth=2, linestyle='-', color='cyan', markersize=20, label='4000s'),
    Line2D([0], [0], linewidth=2, linestyle='--', color='red', markersize=20, label='Close approach'),
    ]

colors = ['darkred', 'red', 'darkorange', 'yellow', 'limegreen']

for j in range(1):
    deviation_lists_ave500 = []

    for i in range(100):
        deviation_lists_ave500.append(np.mean([deviation_lists500[0][i], deviation_lists500[1][i], deviation_lists500[2][i], deviation_lists500[3][i], deviation_lists500[4][i]]))

    deviation_lists_ave1000 = []

    for i in range(100):
        deviation_lists_ave1000.append(np.mean([deviation_lists1000[0][i], deviation_lists1000[1][i], deviation_lists1000[2][i], deviation_lists1000[3][i], deviation_lists1000[4][i]]))

    deviation_lists_ave2000 = []

    for i in range(100):
        deviation_lists_ave2000.append(np.mean([deviation_lists2000[0][i], deviation_lists2000[1][i], deviation_lists2000[2][i], deviation_lists2000[3][i], deviation_lists2000[4][i]]))

    deviation_lists_ave3000 = []

    for i in range(100):
        deviation_lists_ave3000.append(np.mean([deviation_lists3000[0][i], deviation_lists3000[1][i], deviation_lists3000[2][i], deviation_lists3000[3][i], deviation_lists3000[4][i]]))

    deviation_lists_ave4000 = []

    for i in range(100):
        deviation_lists_ave4000.append(np.mean([deviation_lists4000[0][i], deviation_lists4000[1][i], deviation_lists4000[2][i], deviation_lists4000[3][i], deviation_lists4000[4][i]]))

def plot_alldev():
    days = [int(i / 100 * num_days) for i in range(100)]

    plt.figure(figsize=size)

    plt.plot(days, deviation_lists_ave500, linestyle='-', color='fuchsia', linewidth=1, marker='^', ms=3)
    plt.plot(days, deviation_lists_ave1000, linestyle='-', color='blueviolet', linewidth=1, marker='^', ms=3)
    plt.plot(days, deviation_lists_ave2000, linestyle='-', color='blue', linewidth=1, marker='^', ms=3)
    plt.plot(days, deviation_lists_ave3000, linestyle='-', color='dodgerblue', linewidth=1, marker='^', ms=3)
    plt.plot(days, deviation_lists_ave4000, linestyle='-', color='cyan', linewidth=1, marker='^', ms=3)

    plt.ylim(0, 0.9e11)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})
    plt.ylabel("Deviation (10 m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    impact_day2 = 1833
    plt.axvline(x=impact_day2, color='red', linestyle='--', linewidth=1)

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': font_size}

    plt.legend(handles=legend_elements2, prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

def plot_500():
    days = [int(i / 100 * num_days) for i in range(100)]

    plt.figure(figsize=size)

    plt.text(0.77, 0.95, 'Step = 500s', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    for i in range(5):
        plt.plot(days, deviation_lists500[i], linestyle='-', color=colors[i], linewidth=1, marker='o', ms=1.5)

    plt.ylim(0, 0.9e11)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})
    plt.ylabel("Deviation (10 m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    impact_day2 = 1833
    plt.axvline(x=impact_day2, color='red', linestyle='--', linewidth=1)

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': font_size}

    plt.legend(handles=legend_elements1, prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

def plot_1000():
    days = [int(i / 100 * num_days) for i in range(100)]

    plt.figure(figsize=size)

    plt.text(0.77, 0.95, 'Step = 1000s', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    for i in range(5):
        plt.plot(days, deviation_lists1000[i], linestyle='-', color=colors[i], linewidth=1, marker='o', ms=1.5)

    plt.ylim(0, 0.9e11)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})
    plt.ylabel("Deviation (10 m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    impact_day2 = 1833
    plt.axvline(x=impact_day2, color='red', linestyle='--', linewidth=1)

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': font_size}

    plt.legend(handles=legend_elements1, prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

def plot_2000():
    days = [int(i / 100 * num_days) for i in range(100)]

    plt.figure(figsize=size)

    plt.text(0.77, 0.95, 'Step = 2000s', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    for i in range(5):
        plt.plot(days, deviation_lists2000[i], linestyle='-', color=colors[i], linewidth=1, marker='o', ms=1.5)

    plt.ylim(0, 0.9e11)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})
    plt.ylabel("Deviation (10 m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    impact_day2 = 1833
    plt.axvline(x=impact_day2, color='red', linestyle='--', linewidth=1)

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': font_size}

    plt.legend(handles=legend_elements1, prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

def plot_3000():
    days = [int(i / 100 * num_days) for i in range(100)]

    plt.figure(figsize=size)

    plt.text(0.77, 0.95, 'Step = 3000s', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    for i in range(5):
        plt.plot(days, deviation_lists3000[i], linestyle='-', color=colors[i], linewidth=1, marker='o', ms=1.5)

    plt.ylim(0, 0.9e11)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})
    plt.ylabel("Deviation (10 m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    impact_day2 = 1833
    plt.axvline(x=impact_day2, color='red', linestyle='--', linewidth=1)

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': font_size}

    plt.legend(handles=legend_elements1, prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

def plot_4000():
    days = [int(i / 100 * num_days) for i in range(100)]

    plt.figure(figsize=size)

    plt.text(0.77, 0.95, 'Step = 4000s', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    for i in range(5):
        plt.plot(days, deviation_lists4000[i], linestyle='-', color=colors[i], linewidth=1, marker='o', ms=1.5)

    plt.ylim(0, 0.9e11)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel("Time (days)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})
    plt.ylabel("Deviation (10 m)", fontdict={'family': 'DejaVu Serif', 'color': 'black', 'weight': 'normal', 'size': font_size})

    impact_day2 = 1833
    plt.axvline(x=impact_day2, color='red', linestyle='--', linewidth=1)

    legend_font = {'family': 'DejaVu Serif', 'weight': 'normal', 'size': font_size}

    plt.legend(handles=legend_elements1, prop = legend_font, loc = 'upper left')

    plt.tight_layout()

    plt.show()

plot_500()
plot_1000()
plot_2000()
plot_3000()
plot_4000()
plot_alldev()
