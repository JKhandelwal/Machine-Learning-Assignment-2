from sklearn.metrics import confusion_matrix, accuracy_score
import csv
from plot import plot_confusion_matrix


def print_stats(name, Y_pred, Y_test, y_final, initName, key):
    # Stats when running through the results
    print(name)
    print("Confusion Matrix: ")
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)
    plot_confusion_matrix(cm, name, initName, key)
    print('Accuracy Score: %.2f' % accuracy_score(Y_test, Y_pred))
    print()

def write(name, y, y_names):
    # Write the results to a file
    with open("outputs/" + name +".csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(y_names)
        for i in range(0,len(y[0])):
            l_to_write = [x[i] for x in y]
            writer.writerow(l_to_write)
