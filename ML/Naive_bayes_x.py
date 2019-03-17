from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np


def load_datasets():
    return datasets.load_digits(),datasets.load_iris()

def BayesImage(data):

    gnb = GaussianNB()
    n_samples = len(data.images)
    data_images = data.images.reshape((n_samples, -1))
    train_data, test_data, train_label, test_label = train_test_split(data_images, data.target, test_size=0.2)
    clf = gnb.fit(train_data,train_label)

    print(clf.predict(test_data)[:30])
    print(test_label[:30])

def BayesValue(data):
    gnb = GaussianNB()
    train_data, test_data, train_label, test_label = train_test_split(data.data, data.target, test_size=0.2)

    clf = gnb.fit(train_data,train_label)


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

BayesImage(digits)
BayesValue(iris)


# http://hadoopspark.blogspot.com/2016/05/spark-naive-bayes.html