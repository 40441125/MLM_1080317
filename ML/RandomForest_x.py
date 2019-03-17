from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
# import graphviz


def load_datasets():
    return datasets.load_digits(),datasets.load_iris()

def RandomForestImage(data):

    n_samples = len(data.images)
    data_images = data.images.reshape((n_samples, -1))
    train_data, test_data, train_label, test_label = train_test_split(data_images, data.target, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(train_data,train_label)

    print(clf.predict(test_data)[:30])
    print(test_label[:30])

def RandomForestValue(data):

    train_data, test_data, train_label, test_label = train_test_split(data.data, data.target, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(train_data, train_label)
    # dot_data = tree.export_graphviz(clf,out_file=None,
    #                                 feature_names=iris.feature_names,
    #                                 class_names=iris.target_names,
    #                                 filled=True,rounded=True,
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris_DecisionTrees")

    print(clf.predict(test_data)[:30])
    print(test_label[:30])

def show_digits_images(data):
    for i in range(0, 4):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        imside = int(np.sqrt(data.data[i].shape[0]))
        im1 = np.reshape(data.data[i], (imside, imside))
        plt.imshow(im1, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: {}'.format(data.target[i]))
    plt.show()

digits,iris = load_datasets()

RandomForestImage(digits)
RandomForestValue(iris)

# http://hn28082251.blogspot.com/2018/07/random-forest.html