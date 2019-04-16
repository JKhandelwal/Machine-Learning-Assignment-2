from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import math

def logreg(X_train, X_test, Y_train, Y_test, x_final):
    # Linear Regression
    regr = LogisticRegression(solver="lbfgs", max_iter=200)
    regr.fit(X_train, Y_train.values.ravel())
    Y_pred = regr.predict(X_test)
    y_final = regr.predict(x_final)
    return Y_pred, y_final

def tree_classifier(X_train, X_test, Y_train, Y_test, x_final):
    # Decision Tree
    tree_classifier = tree.DecisionTreeClassifier()
    tree_classifier.fit(X_train, Y_train)
    Y_pred = tree_classifier.predict(X_test)
    y_final = tree_classifier.predict(x_final)
    return Y_pred, y_final

def svm(X_train, X_test, Y_train, Y_test, x_final):
    # Support Vector Machines
    SVM = SVC(kernel='rbf')
    SVM.fit(X_train, Y_train.values.ravel())
    Y_pred = SVM.predict(X_test)
    y_final = SVM.predict(x_final)
    return Y_pred, y_final

def rf(X_train, X_test, Y_train, Y_test, x_final):
    # Random Forests
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train.values.ravel())
    Y_pred = rf.predict(X_test)
    y_final = rf.predict(x_final)
    return Y_pred, y_final
