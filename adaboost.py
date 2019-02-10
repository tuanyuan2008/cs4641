from __future__ import division
from sklearn import ensemble, tree
from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

import datasets
import matplotlib.pyplot as plt

"""
Sentiment : trees of various bag_sizes (fixed max_depth=50, n_estimators=10)
"""
MAX_BAG_SIZE = 17499
bag_sizes = range(200, MAX_BAG_SIZE // 2, 200)
MAX_DEPTH = 50
N_ESTIMATORS = 10
train_err = [0] * len(bag_sizes)
test_err = [0] * len(bag_sizes)

for i, b in enumerate(bag_sizes):
    sentiment_data = datasets.sentiment(bag_size=b)

    print('learning a decision tree w/ bag_size=' + str(b))
    t = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=N_ESTIMATORS)
    clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'], clf.predict(sentiment_data['train']['data']))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'], clf.predict(sentiment_data['test']['data']))
    print("Bag\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (b, train_err[i], test_err[i]))

plt.figure()
plt.plot(bag_sizes, test_err, '-', label='test error')
plt.plot(bag_sizes, train_err, '-', label='train error')
plt.title('Boosted Decision Trees: Performance x Bag Size')
plt.xlabel('Bag Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()

# Preparing data with bag_size=4000
sentiment_data = datasets.sentiment(bag_size=4000)

"""
Sentiment : trees of various training set sizes (fixed max_depth=50)
"""
max_n_estimators = range(5, 40, 5)
MAX_DEPTH = 50
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)

for i, e in enumerate(max_n_estimators):
    print('learning a decision tree w/ n_estimators=' + str(e))
    t = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=e)
    clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'], clf.predict(sentiment_data['train']['data']))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'], clf.predict(sentiment_data['test']['data']))
    print("max_nE\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (e, train_err[i], test_err[i]))

plt.figure()
plt.plot(max_n_estimators, test_err, '-', label='test error')
plt.plot(max_n_estimators, train_err, '-', label='train error')
plt.title('Boosted Decision Trees: Performance x Num Estimators')
plt.xlabel('Num Estimators')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()

"""
Sentiment : trees of various bag_sizes (fixed max_depth=50, n_estimators=10)
"""
train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))
MAX_DEPTH = 50
N_ESTIMATORS = 10
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print('training_set_max_size:', train_size, '\n')

for i, o in enumerate(offsets):
    print('learning a decision tree with training_set_size=' + str(o))
    t = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    clf = ensemble.AdaBoostClassifier(base_estimator=t, n_estimators=N_ESTIMATORS)
    clf = clf.fit(sentiment_data['train']['data'][:o], sentiment_data['train']['labels'][:o])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'][:o], clf.predict(sentiment_data['train']['data'][:o]))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'][:o], clf.predict(sentiment_data['test']['data'][:o]))
    print("Offsets\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (o, train_err[i], test_err[i]))

plt.figure()
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.title('Boosted Decision Trees: Performance x Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()
