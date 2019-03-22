from __future__ import division
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import accuracy_score
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

import datasets
import matplotlib.pyplot as plt

"""
Sentiment : neural nets of various bag_sizes (fixed n_hidden=1)
"""
MAX_BAG_SIZE = 17499
bag_sizes = range(200, MAX_BAG_SIZE // 7, 200)
train_err = [0] * len(bag_sizes)
test_err = [0] * len(bag_sizes)

for i, b in enumerate(bag_sizes):
    print('learning a neural net w/ bag_size=' + str(b))

    sentiment_data = datasets.sentiment_nn(bag_size=b)

    inp_len = len(sentiment_data['test']['data'][0])
    out_len = 1
    NET_SHAPE = (inp_len, inp_len // 2, out_len)
    net = buildNetwork(*NET_SHAPE)

    trainer = BackpropTrainer(net, sentiment_data['train_nn'])
    trainer.trainOnDataset(sentiment_data['train_nn'], 10)

    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
            [net.activate(sentiment_data['train']['data'][i]) for k in xrange(len(sentiment_data['train']['data']))])
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
            [net.activate(sentiment_data['test']['data'][i]) for k in xrange(len(sentiment_data['test']['data']))])
    print("Bag\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (b, train_err[i], test_err[i]))

plt.figure()
plt.plot(bag_sizes, test_err, '-', label='test error')
plt.plot(bag_sizes, train_err, '-', label='train error')
plt.title('Neural Nets: Performance x Bag Size')
plt.xlabel('Bag Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()

# Preparing data with bag_size=1500
sentiment_data = datasets.sentiment_nn(bag_size=1500)

"""
Sentiment : neural nets of various n_hidden
"""
inp_len = len(sentiment_data['test']['data'][0])
out_len = 1
net_shapes = [(inp_len, out_len), (inp_len, inp_len//2, out_len),
              (inp_len, inp_len // 2, inp_len // 4, out_len),
              (inp_len, inp_len // 2, inp_len // 4, inp_len // 5, out_len),
              (inp_len, inp_len // 2, inp_len // 4, inp_len // 5, inp_len // 6, out_len)]
train_err = [0] * len(net_shapes)
test_err = [0] * len(net_shapes)

for i, d in enumerate(net_shapes):
    print('learning a neural net w/ net_shape=' + str(d))

    net = buildNetwork(*d)

    trainer = BackpropTrainer(net, sentiment_data['train_nn'])
    trainer.trainOnDataset(sentiment_data['train_nn'], 10)

    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
            [net.activate(sentiment_data['train']['data'][i]) for k in xrange(len(sentiment_data['train']['data']))])
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
            [net.activate(sentiment_data['test']['data'][i]) for k in xrange(len(sentiment_data['test']['data']))])
    print("Shape\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (d, train_err[i], test_err[i]))

plt.figure()
plt.plot(map(lambda x: len(x) - 2, net_shapes), test_err, '-', label='test error')
plt.plot(map(lambda x: len(x) - 2, net_shapes), train_err, '-', label='train error')
plt.title('Neural Nets: Performance x Num Hidden')
plt.xlabel('Num Hidden')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()

"""
Sentiment : neural nets of various training set sizes (fixed n_hidden=1)
"""
sentiment_data = datasets.sentiment(bag_size=1500)

train_size = len(sentiment_data['train']['data'])
offsets = range(int(0.1 * train_size), train_size, int(0.03 * train_size))

inp_len = len(sentiment_data['test']['data'][0])
out_len = 1
NET_SHAPE = (inp_len, inp_len // 2, inp_len // 4, out_len)
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

print('training_set_max_size:', train_size, '\n')

for i, o in enumerate(offsets):
    print('learning a neural net w/ training_set_size=' + str(o))

    sentiment_data = datasets.sentiment_nn(bag_size=1000, offset=o)

    net = buildNetwork(*NET_SHAPE)

    trainer = BackpropTrainer(net, sentiment_data['train_nn'])
    trainer.trainOnDataset(sentiment_data['train_nn'], 10)

    train_err[i] = mean_squared_error(sentiment_data['train']['labels'],
            [net.activate(sentiment_data['train']['data'][i]) for k in xrange(len(sentiment_data['train']['data']))])
    test_err[i] = mean_squared_error(sentiment_data['test']['labels'],
            [net.activate(sentiment_data['test']['data'][i]) for k in xrange(len(sentiment_data['test']['data']))])
    print("Offsets\t\tTrain Err\t\tTest Err")
    print("%.6d\t\t%.6f\t\t%.6f" % (o, train_err[i], test_err[i]))

plt.figure()
plt.plot(offsets, test_err, '-', label='test error')
plt.plot(offsets, train_err, '-', label='train error')
plt.title('Neural Nets: Performance x Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend(loc='lower right')
plt.show()
