from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def scale(train, test):
    # This normalises the Data using the standard Scalar
    scaler = StandardScaler()
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled


def PCA_scaled(train, test):
    pca = PCA()
    pca.fit(train)

    train_scaled = pca.transform(train)
    test_scaled = pca.transform(test)

    return train_scaled, test_scaled

def print_stats(name, Y_pred, Y_test, y_final, initName):
    # Stats when running through the results
    print(name)
    print("Mean squared error: %.2f"% mean_squared_error(Y_test, Y_pred))
    print('Mean Absolute Error: %.2f' % mean_absolute_error(Y_test, Y_pred))
    print('R Squared Result: %.2f' % r2_score(Y_test, Y_pred))

    write(y_final, initName + name.replace(" ",""))

def write(y_final, name):
    # Write the results to a file
    with open("outputs/" + name +".csv", "w") as f:
        for t in y_final:
            f.write(str(t) + "\n")
