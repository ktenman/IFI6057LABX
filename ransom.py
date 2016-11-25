from sklearn import tree
import csv
import random
import matplotlib.pyplot as plt

divided_times = 445
total_run = 100
average = []
for a in range(divided_times):
    average.append([])


def get_passengers(file_name):
    passengers = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            passengers.append(row)
    del passengers[0]
    random.shuffle(passengers)
    return passengers


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


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
    for i in range(total_run):
        passengers = get_passengers("train.csv")
        for i in range(divided_times):
            features, labels = analize_data(passengers, (1 / divided_times + (1 / divided_times) * i))
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(features, labels)
            features, labels = analize_data(passengers, 1)
            score = clf.score(features, labels)
            average[i].append(score)


def print_results():
    x = []
    y = []
    for i in range(100):
        y.append(i+1)
    for i in range(len(average)):
        print ("%d, %f" % (((2*i)+2), (100*mean(average[i]))))
        x.append(100*mean(average[i]))


for i in range(1):
    run()
    print_results()
    print("********************************************************")

# print(clf.predict([[3.0, 1, 34.5]]))

# dot_data = StringIO()
# dot_data = tree.export_graphviz(clf,
#                         out_file=dot_data,
#                 feature_names=features,
#                 class_names=labels,
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data)
# graph.write_pdf("titan.pdf")
