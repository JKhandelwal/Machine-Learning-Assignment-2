import matplotlib as m
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import pandas as pd

def plotStdDev(X, name, key):
    list_unique = X.Y.unique()

    for i in list_unique:
        X0 = X[X["Y"] == i]
        err = []
        x = X0.columns.values[:-1]
        y = []
        for col in X0:
            if col != "Y":
                y.append(np.mean(X0[col]))
                err.append(np.std(X0[col]))
        if i == 0:
            c = 'b'
        elif i ==1:
            c = 'g'
        elif i == 2:
            c = 'r'
        elif i == 3:
            c = 'y'
        elif i == 4:
            c = 'm'

        key_val = key[i]

        plt.errorbar(range(1,769), y, err, linestyle='None', color=c,markerfacecolor=c,  marker='+', capsize=4, label=("Y=" + key_val))
    plt.legend(loc='upper left')
    plt.xlabel("X Feature Number")#
    plt.ylabel("Mean with Standard Deviation")
    plt.title("Mean with Standard Deviation for All features for " + name)
    plt.savefig("plots/stdDev" + str(name) + ".png")
    plt.close()

def plot_fill_between(X, name, key):
    list_unique = X.Y.unique()

    for i in list_unique:
        X0 = X[X["Y"] == i]
        err = []
        x = X0.columns.values[:-1]
        y = []
        for col in X0:
            if col != "Y":
                y.append(np.mean(X0[col]))
                err.append(np.std(X0[col]))
        if i == 0:
            c = 'b'
        elif i ==1:
            c = 'g'
        elif i == 2:
            c = 'r'
        elif i == 3:
            c = 'y'
        elif i == 4:
            c = 'm'

        plt.fill_between(range(1,769), [x + y for x, y in zip(y, err)], [x - y for x, y in zip(y, err)], color=c, label=("Y=" + str(key[i])))

    plt.legend(loc='upper left')
    plt.xlabel("X Feature Number")#
    plt.ylabel("Mean +/- Standard Deviation")
    plt.title("Mean +/- Standard Deviation for All features for " + name)
    plt.savefig("plots/fillBetween" + str(name) + ".png")
    plt.close()

def plot_parallel_coordinates(df, name):

    pc = pd.plotting.parallel_coordinates(df, 'Y', color=('#4286f4', '#f44141'), axvlines=False)

    plt.xlabel("Feature Number")
    plt.ylabel("Sample Number")
    plt.title("Samples through features for Y for " + name)
    plt.savefig("plots/parallel" + str(name) + ".png")
    plt.close()

def plot_p_value(Y, X_values, name):
    plt.scatter(range(1,769), [x[1] for x in Y.values()], marker="+")
    plt.title("P_Value Coefficients for " + name)
    plt.xlabel("X Feature Number")
    plt.ylabel("P Value")
    plt.savefig("plots/PValueplot" + name + ".png")
    plt.close()


def plot_spear(Y, X_values, name):
    plt.scatter(range(1,769), [x[0] for x in Y.values()], marker="+")
    plt.title("Spearman Plot for " + name)
    plt.xlabel("X Feature Number")
    plt.ylabel("Correlation")
    plt.savefig("plots/Spearmanplot" + name + ".png")
    plt.close()

def plot_spearman(df, name):
    Y = df['Y']
    spe = {}
    for i in range(1,769):
        str_val = "X" + str(i)
        spe[str_val] = st.spearmanr(df[str_val], Y)

    X_values = spe.keys()
    plot_spear(spe, X_values, name)
    plot_p_value(spe, X_values, name)

def plot_100(accuracy, iterations, name):
    x = range(1, iterations+1)
    logreg_accuracy = accuracy['logreg']
    rf_accuracy = accuracy['rf']
    svm_accuracy = accuracy['svm']
    tree_accuracy = accuracy['tree']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, logreg_accuracy, c='y', marker="+", label='Logistic Regression')
    ax1.scatter(x, rf_accuracy, c='b', marker="+", label='Random Forests')
    ax1.scatter(x, svm_accuracy, c='r', marker="+", label='Support Vector Machines')
    ax1.scatter(x, tree_accuracy, c='g', marker="+", label='Decision Tree')
    plt.legend();
    plt.xlabel("Iteration Number")
    plt.ylabel("Accuracy Value")
    plt.savefig("plots/AccuracyComparison" + name + ".png")
    plt.close()
