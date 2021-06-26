from os import getcwd, path
from thirdparty import plot_learning_curve
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import argparse

def build_names(names_path):
    names = []

    with open(names_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            split = line.split(":")
            if line[0] != "|" and len(split) == 2:
                names.append(split[0])

    return names + ["is_ad"]

def build_unknown_indices(data_path):
    indices = []

    i = 0
    with open(data_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            if "?" in line:
                indices.append(i)
            i += 1

    return indices

def possible_unknown(val):
    return -1 if "?" in val else float(val)

def is_ad(val):
    return 1 if val == "ad." else 0

def avg(l):
    return sum(l) / len(l)

def get_data(data_path, names_path, drop_unknowns):
    names = build_names(names_path)
    convertes = {
        "height": possible_unknown, "width": possible_unknown, "aratio": possible_unknown,
        "local": possible_unknown, "is_ad": is_ad
    }

    unknown_indices = build_unknown_indices(data_path) if drop_unknowns else []
    dataframe = read_csv(data_path, names=names, converters=convertes, skiprows=unknown_indices)

    X = dataframe.values[:,0:len(names)-1]
    Y = dataframe.values[:,len(names)-1]

    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, Y)

    return dataframe, dict(zip(names, model.feature_importances_.tolist()))

def main(importance_threshold, drop_unknowns):
    data_path = path.join(getcwd(), "data", "ad.data")
    names_path = path.join(getcwd(), "data", "ad.names")
    dataframe, importances = get_data(data_path, names_path, drop_unknowns)

    unimportant_features = [k for k, v in importances.items() if v < importance_threshold]
    important_features = {k: v for k, v in importances.items() if v >= importance_threshold}
    dataframe.drop(unimportant_features, axis=1, inplace=True)

    print(f"Unkown features {'' if drop_unknowns else 'not '}dropped", flush=True)
    print(f"Working on {len(dataframe)} records", flush=True)
    print(f"Importance threshold: {importance_threshold}, average: {avg(importances.values())}", flush=True)
    print(f"Dropped {len(unimportant_features)} features, using {len(important_features)}:", flush=True)

    for feature, importance in important_features.items():
        print(f"- {feature}: {importance:.6f}")

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
        plot_learning_curve(clf, name, X, Y.values.ravel(), axes=axes[:,i], ylim=(0.8, 1.01), cv=cv, n_jobs=4)

    fig.tight_layout()
    plt.savefig(f"learning_curves-du{drop_unknowns}-it{importance_threshold}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--importance-threshold", type=float, default=0.01)
    parser.add_argument("-d", "--drop-unknowns", action="store_true", default=False)
    args = parser.parse_args()

    main(args.importance_threshold, args.drop_unknowns)
