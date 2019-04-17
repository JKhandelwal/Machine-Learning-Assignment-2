import pandas as pd
from sklearn.model_selection import train_test_split
from plot import *
from utils import *
from models import *
from statistics import mean
import numpy as np
import argparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import json
import ast

def prepare(df):
    if df.isnull().values.any():
        print("There are Null values")
        df.dropna()

    df = df.astype('float64')

def read_data(name):
    print("-----------Runnning " + name + "-----------")
    # Cleaning the data, removing nulls if any, and converting all
    # values to floats for easy scalar transforms
    x_df = pd.read_csv(name + '/X.csv', header=None)
    y_df = pd.read_csv(name + '/y.csv', header=None)
    x_df.columns = ["X" + str(x) for x in range(1,len(x_df.columns) + 1)]
    y_df.columns = ["Y"]
    x_to_classify = pd.read_csv(name + '/XToClassify.csv', header=None)
    with open (name + "/key.txt") as f:
        str_key = f.read()
        key = ast.literal_eval(str_key)
    prepare(x_df)
    prepare(y_df)
    prepare(x_to_classify)

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(x_df, y_df.values.ravel())
    model = SelectFromModel(clf, prefit=True)
    x_new = model.transform(x_df)
    x_to_classify_new = model.transform(x_to_classify)

    print("Some Features Selected")
    split(x_new, y_df, x_to_classify_new, "subset_" + name , False, key)

    print("All Features")
    split(x_df, y_df, x_to_classify, "all_" + name, True, key)

def split(X, Y, x_to_classify, name, call_plot, key):

    if iteration:
        perform_iteration(X, Y, 100, x_to_classify, name)
        return

    # 70/30 split for training and testing data,
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,
                                                        test_size=0.3, stratify=Y)


    # Visualise the Training Data
    if call_plot:
        plot(X_train, Y_train, name, key)
    models(X_train, X_test, Y_train, Y_test, x_to_classify, name)

def models(X_train, X_test, Y_train, Y_test, x_final, name):
    # Run the algorithms and print the result.
    y = []
    y_names = []
    Y_pred, y_final = logreg(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("Logistic Regression", Y_pred, Y_test, y_final, name)
    y.append(y_final)
    y_names.append("Logistic Regression")

    Y_pred, y_final = tree_classifier(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("Decision Tree", Y_pred, Y_test, y_final, name)
    y.append(y_final)
    y_names.append("Decision Tree")

    Y_pred, y_final = rf(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("Random Forests", Y_pred, Y_test, y_final, name)
    y.append(y_final)
    y_names.append("Random Forests")

    Y_pred, y_final = svm(X_train, X_test, Y_train, Y_test, x_final)
    print_stats("SVM", Y_pred, Y_test, y_final, name)
    y.append(y_final)
    y_names.append("SVM")

    write(name, y, y_names)


def plot(X, Y, name, key):
    # Run all of the plots
    df = pd.concat([X,Y], axis=1)
    plotStdDev(df, name, key)
    plot_spearman(df, name)
    plot_fill_between(df, name, key)
    plot_parallel_coordinates(df, name)


def perform_iteration(X, Y, iterations, x_final, name):
        # For 100 Iterations
        accuracy = {}
        accuracy['logreg'] = []
        accuracy['rf'] = []
        accuracy['svm'] = []
        accuracy['tree'] = []
        for i in range(0, iterations):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            Y_pred, y_final = logreg(X_train, X_test, Y_train, Y_test, x_final)
            accuracy['logreg'].append(accuracy_score(Y_test, Y_pred))

            Y_pred, y_final = tree_classifier(X_train, X_test, Y_train, Y_test, x_final)
            accuracy['tree'].append(accuracy_score(Y_test, Y_pred))

            Y_pred, y_final = rf(X_train, X_test, Y_train, Y_test, x_final)
            accuracy['rf'].append(accuracy_score(Y_test, Y_pred))

            Y_pred, y_final = svm(X_train, X_test, Y_train, Y_test, x_final)
            accuracy['svm'].append(accuracy_score(Y_test, Y_pred))

        plot_100(accuracy, iterations, name)

        print("Logistic Regression Accuracy Score Standard deviation: " + str(np.std(accuracy['logreg'])))
        print("Random Forests Accuracy Score Standard deviation: " + str(np.std(accuracy['rf'])))
        print("Support Vector Machines Accuracy Score Standard deviation: " + str(np.std(accuracy['svm'])))
        print("Decision Tree Accuracy Score Standard deviation: " + str(np.std(accuracy['tree'])))

        print("Logistic Regression Accuracy Mean: " + str(mean(accuracy['logreg'])))
        print("Random Forests Accuracy Mean: " + str(mean(accuracy['rf'])))
        print("Support Vector Machines Accuracy Mean: " + str(mean(accuracy['svm'])))
        print("Decision Tree Accuracy Mean: " + str(mean(accuracy['tree'])))


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
    read_data("binary")
    read_data("multiclass")
