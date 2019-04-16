import matplotlib as m
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

def plotStdDev(X):
    X0 = X[X["Y"] == 0]
    X1 = X[X["Y"] == 1]
    err0 = []
    x0 = X0.columns.values[:-1]
    y0 = []
    for col in X:
        if col != "Y":
            y0.append(np.mean(X0[col]))
            err0.append(np.std(X0[col]))
    err1 = []
    x1 = X1.columns.values[:-1]
    y1 = []
    for col in x1:
        if col != "Y":
            y1.append(np.mean(X1[col]))
            err1.append(np.std(X1[col]))

    plt.errorbar(range(1,769), y0, err0, linestyle='None', color='g',markerfacecolor='g',  marker='+', capsize=4, label="Y=0")
    plt.errorbar(range(1,769), y1, err1, linestyle='None', color='b',markerfacecolor='b',  marker='+',capsize=4, label="Y=1")
    plt.legend(loc='upper left')
    plt.savefig("plots/stdDev.png")
    plt.close()
    plt.errorbar(range(1,769), y1, err1, linestyle='None', color='b',markerfacecolor='b',  marker='+',capsize=4, label="Y=1")
    plt.legend(loc='upper left')
    plt.savefig("plots/stdDev1.png")
    plt.close()
    plt.errorbar(range(1,769), y0, err0, linestyle='None', color='g',markerfacecolor='g',  marker='+', capsize=4, label="Y=0")
    plt.legend(loc='upper left')
    plt.savefig("plots/stdDev0.png")
    plt.close()

def plot_p_value(Y, X_values):
    plt.scatter(range(1,769), [x[1] for x in Y.values()], marker="+")
    plt.title("P_Value Coefficients of ")
    plt.xlabel("X Feature Number")
    plt.ylabel("P Value")
    plt.savefig("plots/PValueplot.png")
    plt.close()


def plot_spear(Y, X_values):
    plt.scatter(range(1,769), [x[0] for x in Y.values()], marker="+")
    plt.title("Spearman Plot")
    plt.xlabel("X Feature Number")
    plt.ylabel("Correlation")
    plt.savefig("plots/Spearmanplot.png")
    plt.close()

def plot_spearman(df):
    Y = df['Y']
    spe = {}
    for i in range(1,769):
        str_val = "X" + str(i)
        spe[str_val] = st.spearmanr(df[str_val], Y)

    X_values = spe.keys()
    plot_spear(spe, X_values)
    plot_p_value(spe, X_values)

def plot_100(mse, r, iterations):
    x = range(1,iterations+1)
    lin_mse = mse['lin']
    ridge_mse = mse['ridge']
    lasso_mse = mse['lasso']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, lin_mse, c='b', marker="s", label='Linear Regression')
    ax1.scatter(x, ridge_mse, c='r', marker="o", label='Ridge Regression')
    ax1.scatter(x, lasso_mse, c='g', marker="+", label='Lasso Regression')
    plt.legend();
    plt.xlabel("Iteration Number")
    plt.ylabel("MSE Value")
    plt.savefig("plots/MSEComparison.png")
    plt.close()

    lin_r =  r['lin']
    ridge_r = r['ridge']
    lasso_r = r['lasso']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, lin_r, c='b', label='Linear Regression')
    ax1.scatter(x, ridge_r, c='r', label='Ridge Regression')
    ax1.scatter(x, lasso_r, c='g', label='Lasso Regression')
    plt.legend();
    plt.title("R Squared Comparisons")
    plt.xlabel("Iteration Number")
    plt.ylabel("R^2 Value")
    plt.savefig("plots/R2Comparison.png")
    plt.close()


def plot_hist(X):
    i = 0
    for j in range(1,11):
        if j < 9:
            str_val = "X" + str(j)
        else:
            str_val = "Y" + str(j-8)

        plt.hist(X[str_val],edgecolor='black', linewidth=1.2)
        plt.title("Hist of " + str_val)
        plt.xlabel(str_val)
        plt.ylabel("Frequency")
        plt.savefig("plots/hist" + str_val +".png")
        plt.close()

def plot_XY(df):
    for i in range(1, 3):
        for j in range(1,9):
            x = "X" + str(j)
            y = "Y" + str(i)
            plt.scatter(df[x], df[y])
            plt.title("Plot of " + x +  " against " + y)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.savefig("plots/plot" + x + "_" + y + ".png")
            plt.close()
