import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plot import *
from utils import *
from models import *
from statistics import mean
import numpy as np
import argparse

def prepare(df):
    df.dropna()
    if df.isnull().values.any():
        print("There are Null values")
        df.dropna()
    # Convert to Float 64 as to not have to convert implicitly for scaling
    df = df.astype('float64')

def read_data():
    # Cleaning the data, removing nulls if any, and converting all
    # values to floats for easy scalar transforms
    x_df = pd.read_csv("binary/X.csv")
    y_df = pd.read_csv("binary/y.csv")
    x_df.columns = ["X" + str(x) for x in range(1,len(x_df.columns) + 1)]
    y_df.columns = ["Y"]
    x_to_classify = pd.read_csv("binary/xToClassify.csv")
    prepare(x_df)
    prepare(y_df)
    prepare(x_to_classify)

    split(x_df, y_df, x_to_classify)

def split(X, Y, x_final):

    # 60/40 split for training and testing data,
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.3, stratify=Y)
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    # for train_index, test_index in sss.split(X, Y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]

    # Visualise the Training Data
    plot(X_train, Y_train)
    # Scale the Data by sclaing and polynomial fitting
    X_train, X_test = scale(X_train, X_test)
    print("NON PCA")
    models(X_train, X_test, Y_train, Y_test, x_final)

    X_train, X_test = PCA_scaled(X_train, X_test)
    print("PCA")
    # models(X_train, X_test, Y_train, Y_test, x_final)

def models(X_train, X_test, Y_train, Y_test, x_final):
    # Run the Three algorithms and print the result.

    Y_pred, y_final = logreg(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("Logistic Regression", Y_pred, Y_test, y_final, "binary")

    Y_pred, y_final = tree_classifier(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("Tree", Y_pred, Y_test, y_final, "binary")

    Y_pred, y_final = tree_classifier(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("Random Forests", Y_pred, Y_test, y_final, "binary")

    Y_pred, y_final = svm(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("SVM", Y_pred, Y_test, y_final, "binary")


def plot(X, Y):
    # Run all of the plots
    df = pd.concat([X,Y], axis=1)
    plotStdDev(df)
    plot_spearman(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iteration",action="store_true", help="Do 100 iterations")
    args = parser.parse_args()
    if args.iteration:
        print("Iterations being done")
        iteration = True
    else:
        iteration = False
        print("Normal usage")
    read_data()
