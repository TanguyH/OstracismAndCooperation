import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
#import pandas as pd
plt.style.use('classic')
#sns.set_style("whitegrid", {'grid.linestyle': '--'})
WIDTH = 40.
COLOR_X = "#023AFF"
COLOR_Y = "#FF0E2B"
COLOR_Z = "#FFEC00"
COLOR_V = "#FF9900"
COLOR_W = "#00FFFC"

#COLOR_V = "#D60EFF"
#COLOR_W = "#85FF00"

DATA_PATH = "re_data/"

COOPERATE = "X"
DEFECT = "Y"
ABSTAIN = "Z"
POOLPUNISH = "V'"
PEERPUNISH = "W'"

STRATEGIES = [COOPERATE, DEFECT, ABSTAIN, POOLPUNISH, PEERPUNISH]
COLORS = [COLOR_X, COLOR_Y, COLOR_Z, COLOR_V, COLOR_W]

NO_PUNISHMENT = "no_punishment"
POOL_PUNISHMENT = "pool"
PEER_PUNISHMENT = "peer"
COMPETITION = "competition"

COMPULSORY = "compulsory"
VOLUNTARY = "voluntary"
SECOND_ORDER = "SOP"

def determineAvailableStrategies(case, participation = COMPULSORY):
    available_strategies = [COOPERATE, DEFECT]
    if participation == VOLUNTARY:
        available_strategies.append(ABSTAIN)
    if case in [POOL_PUNISHMENT, COMPETITION]:
        available_strategies.append(POOLPUNISH)
    if case in [PEER_PUNISHMENT, COMPETITION]:
        available_strategies.append(PEERPUNISH)
    return available_strategies

def plotData(ax, data, available_strategies, param="s"):
    for i in range(1, len(data)):
        if param in ["timec", "timeu"]:
            strategy = available_strategies[i//len(available_strategies)-1]
        else:
            strategy = available_strategies[i-1]

        x_axis = data[0]
        if param in ["s", "timec", "timeu", "M", "alpha"]:
            rem_ind = []
            for j, elem in enumerate(data[i]):
                if np.isnan(elem):
                    rem_ind.append(j)
                elif np.isinf(elem):
                    rem_ind.append(j)

            for j in range(len(rem_ind)-1, -1, -1):
                index = rem_ind[j]
                x_axis = x_axis[:index] + x_axis[index+1:]

            data[i] = list(filter(lambda a: not np.isinf(a) and not np.isnan(a), data[i]))
            ax.plot(x_axis, data[i], label="${}$".format(strategy), c=COLORS[STRATEGIES.index(strategy)], linewidth=1.5)
        else:
            ax.scatter(data[0], data[i], label="${}$".format(strategy), c=COLORS[STRATEGIES.index(strategy)], linewidth=1, marker="$\Delta$", edgecolors=COLORS[STRATEGIES.index(strategy)], s=25)#marker="D","$\Delta$", marker="h"

def plotCase(case, participation, position = 121, param = "s", order="", solo=False):
    available_strategies = determineAvailableStrategies(case, participation)

    # define file naming characteristics
    prefix = ""
    if param in ["timec", "timeu"]:
        subfile = "times"
        if param == "timec":
            prefix = "conditional_"
        else:
            prefix = "unconditional_"
    else:
        subfile = param
        if order == SECOND_ORDER:
            prefix = "SOP_"

    # open file and declare reader
    csvfile = open("{}{}/{}{}_{}.csv".format(DATA_PATH, subfile, prefix, case, participation))
    reader = csv.DictReader(csvfile)

    if param in ["timec","timeu"]:
        data = [[] for i in range(len(available_strategies) * (len(available_strategies)-1) + 1)]

        for row in reader:
            data[0].append(np.longdouble(row["s"]))
            for i, resident in enumerate(available_strategies):
                remaining_strategies = available_strategies[:i] + available_strategies[i+1:]
                for j, invader in enumerate(remaining_strategies):
                    if resident != invader:
                        data[i * len(remaining_strategies) + j +1].append(np.longdouble(row["{}{}".format(resident, invader)]))
    else:
        data = [[] for i in range(len(available_strategies) + 1)]

        for row in reader:
            data[0].append(np.longdouble(row[param]))
            for i, strategy in enumerate(available_strategies):
                data[i+1].append(np.longdouble(row[strategy]))

    if not solo:
        ax = plt.subplot(position)
    else:
        ax = plt.subplot()
    plotData(ax, data, available_strategies, param)

    if param == "M":
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlim([5, 100])
    elif param == "alpha":
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlim([-.02, 1.27])
    else:
        ax.set_xscale("log", nonposx='clip')
        if param == "mu":
            ax.set_xlim([0., 1.15])
            ax.set_ylim([-0.02, 1.02])
        else:
            if param == "s":
                ax.set_ylim([-0.02, 1.02])
                ax.set_xlim([10**-4, 10**4])
            else:
                ax.set_xlim([0.0, 10000])
            #else:
                #ax.set_ylim([-0.02, 3500])

    plt.grid(True)
    if param in ["s","timec", "timeu"]:
        ticks = [10**-4, 10**-2, 10**0, 10**2, 10**4]
        labels = [0.0001, 0.01, 1, 100, 10000]
        xlabel = "Imitation strength s"
    elif param == "mu":
        ticks = [10**-4, 10**-3, 10**-2, 10**-1, 10**0]
        labels = [0.0001, 0.001, 0.01, 0.1 ,1]
        xlabel = "Mutation rate $\mu$"
    elif param == "alpha":
        ticks = [0, .25, .5, .75 ,1, 1.25]
        labels = [0, .25, .5, .75 ,1, 1.25]
        xlabel = "Reimbursement rate $\ alpha$"
    else:
        ticks = [5, 100, 200, 300, 400, 500]
        labels = [5, 100, 200, 300, 400, 500]
        xlabel = "Population size M"

    if position == 121:
        plt.ylabel("Frequency", fontsize=20, labelpad=20)
    plt.xticks(ticks, labels)
    plt.xlabel(xlabel, fontsize=20, labelpad=10)

    return ax

def plotGraphs(case, param="s", order="", solo = False, participation = COMPULSORY):
    fig = plt.figure()

    """
    if dual:
        ax = plotCase(case, VOLUNTARY, 121, param, order)
        ax = plotCase(dual_case, VOLUNTARY, 122, param, dual_order)
    else:
    """

    if not solo:
        ax = plotCase(case, COMPULSORY, 121, param, order)
        ax = plotCase(case, VOLUNTARY, 122, param, order)
    else:
        ax = plotCase(case, participation, 121, param, order, solo = solo)

    max_strategies = len(determineAvailableStrategies(case, VOLUNTARY))

    """
    if dual_case:
        max_strategies = max(max_strategies, len(determineAvailableStrategies(dual_case, VOLUNTARY)))
    """
    handles, labels = ax.get_legend_handles_labels()

    if param == "mu":
        fig.legend(handles, labels, loc='upper center', ncol=max_strategies, framealpha=0., fontsize=18, borderaxespad=-.375)
    else:
        if not solo:
            width = WIDTH
        else:
            width = WIDTH / 2
            if max_strategies == 5:
                width /= 1.35
        fig.legend(handles, labels, loc='upper center', ncol=max_strategies, framealpha=0., labelspacing=5., handlelength=(width/(1.3*max_strategies)), fontsize=18, borderaxespad=-.375) # prop={'size': 16}, mode='expand'
    plt.show()

if __name__ == "__main__":
    fig_size = [12., 5.5]
    #fig_size = [7.,6.] # for solo
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'figure.subplot.bottom' : 0.15})


    #plotGraphs(COMPETITION, "M", "", solo = True, participation = VOLUNTARY)
    plotGraphs(COMPETITION, "alpha", SECOND_ORDER, solo = False)#, participation = VOLUNTARY)
