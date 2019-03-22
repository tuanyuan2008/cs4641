from __future__ import division
from sklearn import svm
from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

import datasets
import matplotlib.pyplot as plt

"""
Sentiment : SVMs of various bag_sizes (fixed degree=3)
"""
MAX_BAG_SIZE = 17499
bag_sizes = range(200, MAX_BAG_SIZE // 2, 200)
DEGREE = 3
train_err = [0] * len(bag_sizes)
test_err = [0] * len(bag_sizes)

for i, b in enumerate(bag_sizes):
    sentiment_data = datasets.sentiment(bag_size=b)

    print('learning an SVM w/ bag_size=' + str(b))
    clf = svm.SVC(degree=DEGREE)
    clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'], clf.predict(sentiment_data['train']['data']))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'], clf.predict(sentiment_data['test']['data']))
    print("Bag\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (b, train_err[i], test_err[i]))

plt.figure()
plt.plot(bag_sizes, test_err, '-', label='test error')
plt.plot(bag_sizes, train_err, '-', label='train error')
plt.title('SVM: Performance x Bag Size')
plt.xlabel('Bag Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()

# Preparing data with bag_size=2000
sentiment_data = datasets.sentiment(bag_size=2000)

"""
Sentiment : SVM of various degrees
"""
degrees = range(6)
train_err = [0] * len(degrees)
test_err = [0] * len(degrees)

for i, d in enumerate(degrees):
    print('learning an SVM w/ degree=' + str(d))
    clf = svm.SVC(kernel='poly', degree=d)
    clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'], clf.predict(sentiment_data['train']['data']))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'], clf.predict(sentiment_data['test']['data']))
    print("Degree\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (d, train_err[i], test_err[i]))

plt.figure()
plt.plot(degrees, test_err, '-', label='test error')
plt.plot(degrees, train_err, '-', label='train error')
plt.title('SVM: Performance x Degree')
plt.xlabel('Degree')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()

"""
Sentiment : SVMs of various training set sizes (fixed degree=1)
"""
train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))
MAX_DEPTH = 35
DEGREE = 1
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print('training_set_max_size:', train_size, '\n')

for i, o in enumerate(offsets):
    print('learning an SVM w/ training_set_size=' + str(o))
    clf = svm.SVC(kernel = 'linear',degree=DEGREE)
    clf = clf.fit(sentiment_data['train']['data'][:o], sentiment_data['train']['labels'][:o])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'][:o], clf.predict(sentiment_data['train']['data'][:o]))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'][:o], clf.predict(sentiment_data['test']['data'][:o]))
    print("Offsets\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (o, train_err[i], test_err[i]))

plt.figure()
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.title('SVMs: Performance x Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()
