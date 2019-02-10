from __future__ import division
from sklearn import neighbors
from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

import datasets
import matplotlib.pyplot as plt

"""
Sentiment : kNN of various bag_sizes (fixed n_estimators=5)
"""
# MAX_BAG_SIZE = 17499
# bag_sizes = range(200, MAX_BAG_SIZE // 2, 200)
# N_NEIGHBORS = 5
# train_err = [0] * len(bag_sizes)
# test_err = [0] * len(bag_sizes)

# for i, b in enumerate(bag_sizes):
#     sentiment_data = datasets.sentiment(bag_size=b)

#     print('learning a kNN classifier w/ bag_size=' + str(b))
#     clf = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS78u)
#     clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

#     train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'], clf.predict(sentiment_data['train']['data']))
#     test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'], clf.predict(sentiment_data['test']['data']))
#     print("Bag\t\tTrain Err\t\tTest Err")
#     print("%.6d\t\t%.6f\t\t%.6f" % (b, train_err[i], test_err[i]))

# plt.figure()
# plt.plot(bag_sizes, test_err, '-', label='test error')
# plt.plot(bag_sizes, train_err, '-', label='train error')
# plt.title('KNN Classifier: Performance x Bag Size')
# plt.xlabel('Bag Size')
# plt.ylabel('Error')
# plt.legend(loc='lower right')
# plt.show()

# Preparing data with bag_size=2500
sentiment_data = datasets.sentiment(bag_size=2500)

"""
Sentiment : kNN of various k's
"""
# ks = range(3, 8)
# train_err = [0] * len(ks)
# test_err = [0] * len(ks)

# for i, k in enumerate(ks):
#     print('learning a kNN classifier w/ n_neighbors=' + str(k))
#     clf = neighbors.KNeighborsClassifier(n_neighbors=k)
#     clf = clf.fit(sentiment_data['train']['data'], sentiment_data['train']['labels'])

#     train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'], clf.predict(sentiment_data['train']['data']))
#     test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'], clf.predict(sentiment_data['test']['data']))
#     print("Neigh #\t\tTrain Err\t\tTest Err")
#     print("%.6d\t\t%.6f\t\t%.6f" % (k, train_err[i], test_err[i]))

# plt.figure()
# plt.plot(ks, test_err, '-', label='test error')
# plt.plot(ks, train_err, '-', label='train error')
# plt.title('KNN Classifier: Performance x K')
# plt.xlabel('K')
# plt.ylabel('Error')
# plt.legend(loc='lower right')
# plt.show()

"""
Sentiment : kNN of various training set sizes (fixed n_neighbors=3)
"""
train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))
N_NEIGHBORS = 3
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print('training_set_max_size:', train_size, '\n')

for i, o in enumerate(offsets):
    print('learning a kNN classifier with training_set_size=' + str(o))
    clf = neighbors.KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    clf = clf.fit(sentiment_data['train']['data'][:o], sentiment_data['train']['labels'][:o])

    train_err[i] = 1 - accuracy_score(sentiment_data['train']['labels'][:o], clf.predict(sentiment_data['train']['data'][:o]))
    test_err[i] = 1 - accuracy_score(sentiment_data['test']['labels'][:o], clf.predict(sentiment_data['test']['data'][:o]))
    print("Offsets\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (o, train_err[i], test_err[i]))

plt.figure()
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.title('KNN CLassifier: Performance x Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()