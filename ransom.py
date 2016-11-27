from sklearn import tree
import random
import csv
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

start_time = time.time()

divided_times = 100
tree_score_list = []
set_size_list = []
forest_score_list = []
neighbors_score_list = []


def get_passengers(file_name):
    passengers = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            passengers.append(row)
    del passengers[0]
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
    set_size = int(len(passengers) * limit)
    for i in range(set_size):
        passenger = passengers[i]
        temp = [
            float(passenger[2]),  # REISIJA KLASS
            dict.get(passenger[4]),  # SUGU
        ]
        try:
            # temp.append(float(passenger[5]))  # VANUS
            temp.append(int(passenger[6]))  # ÕED/VENNAD/ABIKAASED
            # temp.append(float(passenger[7]))  # VANEMAD/LAPSED
            labels.append(passenger[1])
            features.append(temp)
        except ValueError:
            pass
    return features, labels, set_size


def run():
    passengers = get_passengers("train.csv")
    for i in range(divided_times):
        train_features, train_labels, set_size = analize_data(passengers, (1 / divided_times + (1 / divided_times) * i))

        # clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(train_features, train_labels)

        # forest = RandomForestClassifier(n_estimators=10)
        # forest = forest.fit(train_features, train_labels)


        neighbors = KNeighborsClassifier(3)
        neighbors = neighbors.fit(train_features, train_labels)

        test_features, test_labels, temp = analize_data(get_passengers("IFI6057_hw2016_test.csv"), 1)

        # tree_score = clf.score(test_features, test_labels)
        # forest_score = forest.score(test_features, test_labels)

        neighbors_score = neighbors.score(test_features, test_labels)

        # tree_score_list.append(tree_score * 100)
        # forest_score_list.append(forest_score * 100)
        neighbors_score_list.append(neighbors_score * 100)

        set_size_list.append(set_size)


def print_results(list_name, list):
    print("--- %s seconds ---" % round(time.time() - start_time, 2))
    print()
    print("MIN: %f (set size: %d)" % (
        min(list),
        set_size_list[list.index(min(list))],
    ))
    print("MAX: %f (set size: %d)" % (
        max(list),
        set_size_list[list.index(max(list))]
    ))
    for i in range(len(neighbors_score_list)):
        # print("Tree: %d; %f" % (set_size_list[i], tree_score_list[i]))
        # print("Forest: %d; %f" % (set_size_list[i], forest_score_list[i]))
        print("%s: %d; %f" % (list_name, set_size_list[i], list[i]))


def show_results():
    # plt.xlabel('Test data set size')
    # plt.ylabel('Accuracy')
    # plt.title('Decision Tree Classifier')
    # plt.grid(True)
    # plt.plot(set_size_list, tree_score_list, color='b',
    #          linewidth=1.0)
    # plt.axis([0, 200, 0, 100])
    # plt.show()
    #
    # plt.xlabel('Test data set size')
    # plt.ylabel('Accuracy')
    # plt.title('Decision Tree Classifier')
    # plt.grid(True)
    # plt.plot(set_size_list, tree_score_list, color='b',
    #          linewidth=1.0)
    #
    # plt.show()


    #######################################
    # plt2.xlabel('Test data set size')
    # plt2.ylabel('Accuracy')
    # plt2.title('Forest Classifier')
    # plt2.grid(True)
    # plt2.plot(set_size_list, forest_score_list, color='g',
    #           linewidth=1.0)
    # plt2.show()
    #
    # plt2.xlabel('Test data set size')
    # plt2.ylabel('Accuracy')
    # plt2.title('Forest Classifier')
    # plt2.grid(True)
    # plt2.plot(set_size_list, forest_score_list, color='g',
    #           linewidth=1.0)
    # plt2.axis([0, 200, 0, 100])
    # plt2.show()
    #######################################
    plt3.xlabel('Test data set size')
    plt3.ylabel('Accuracy')
    plt3.title('K-Nearest Neighbors Classifier')
    plt3.grid(True)
    plt3.plot(set_size_list, neighbors_score_list, color='y',
              linewidth=1.0)
    plt3.axis([0, 200, 0, 100])
    plt3.show()

    plt3.xlabel('Test data set size')
    plt3.ylabel('Accuracy')
    plt3.title('K-Nearest Neighbors Classifier')
    plt3.grid(True)
    plt3.plot(set_size_list, neighbors_score_list, color='y',
              linewidth=1.0)
    plt3.show()


run()
print_results("Neighbors", neighbors_score_list)
show_results()
