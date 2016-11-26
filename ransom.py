from sklearn import tree
import scipy
import random
import csv

divided_times = 891
score_list = []


def get_passengers(file_name):
    passengers = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            passengers.append(row)
    del passengers[0]
    random.shuffle(passengers)
    return passengers


def analize_data(passengers, limit):
    labels = []
    features = []
    dict = {
        'male': 0,
        'female': 1,
        'Q': 2,
        'S': 3,
        'C': 4
    }
    till = int(len(passengers) * limit)
    for i in range(till):
        element = passengers[i]
        abi = [
            float(element[2]),
            dict.get(element[4]),
        ]
        try:
            # abi.append(float(element[5]))
            abi.append(int(element[6]))
            abi.append(float(element[7]))
            abi.append(float(0 if dict.get(element[11]) is None else dict.get(element[11]))),
            labels.append(element[1])
            features.append(abi)
        except ValueError:
            pass
    return features, labels


def run():
    passengers = get_passengers("train.csv")
    for i in range(divided_times):
        features, labels = analize_data(passengers, (1 / divided_times + (1 / divided_times) * i))
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(features, labels)
        features, labels = analize_data(get_passengers("IFI6057_hw2016_test.csv"), 1)
        score = clf.score(features, labels)
        score_list.append(score)


def print_results():
    for i in range(len(score_list)):
        print("%d, %f" % (i + 1), (score_list[i]))


run()
print_results()
