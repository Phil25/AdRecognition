from os import getcwd, path
from thirdparty import plot_learning_curve
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

IMPORTANCE_THRESHOLD = 0.01

def build_names(names_path):
    names = []

    with open(names_path, "r") as f:
        for line in f:
            split = line.split(":")
            if line[0] != "|" and len(split) == 2:
                names.append(split[0])

    return names + ["is_ad"]

def possible_unknown(val):
    return -1 if "?" in val else float(val)

def is_ad(val):
    return 1 if val == "ad." else 0

def get_data(data_path, names_path):
    names = build_names(names_path)
    convertes = {
        "height": possible_unknown, "width": possible_unknown, "aratio": possible_unknown,
        "local": possible_unknown, "is_ad": is_ad
    }

    dataframe = read_csv(data_path, names=names, converters=convertes)

    X = dataframe.values[:,0:len(names)-1]
    Y = dataframe.values[:,len(names)-1]

    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, Y)

    return dataframe, dict(zip(names, model.feature_importances_.tolist()))

def main():
    data_path = path.join(getcwd(), "data", "ad.data")
    names_path = path.join(getcwd(), "data", "ad.names")
    dataframe, importances = get_data(data_path, names_path)

    unimportant_features = [k for k, v in importances.items() if v < IMPORTANCE_THRESHOLD]
    dataframe.drop(unimportant_features, axis=1, inplace=True)

    X = dataframe.iloc[:,:-1] # features
    Y = dataframe.iloc[:,-1:] # class

    classifiers = [
        (RandomForestClassifier(), "Random Forest"),
        (AdaBoostClassifier(), "AdaBoost"),
        (SVC(kernel="linear", C=0.025), "Linear SVM"),
        (SVC(gamma=2, C=1), "RBF SVM"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    for i in range(len(classifiers)):
        clf, name = classifiers[i]
        plot_learning_curve(clf, name, X, Y.values.ravel(), axes=axes[:,i], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    fig.tight_layout()
    plt.savefig("learning_curves.png")

if __name__ == "__main__":
    main()