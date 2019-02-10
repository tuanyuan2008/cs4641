from collections import Counter

import random
import numpy as np
from sklearn import preprocessing
from pybrain.datasets.classification import ClassificationDataSet

def shuffle_data(data, labels, ten_percent=False):
    z = zip(data, labels)
    random.shuffle(z)
    if ten_percent:
        z = z[:int(len(z) * 0.1)]
    return zip(*z)


def read_sentiment_data(f_name):
    data = []
    with open('datasets/sentiment-analysis/' + f_name, 'rb') as f:
        review = ''
        for line in f:
            if line[-2] in ['0', '1']:
                data.append((review + line[:-3], line[-2]))
                review = ''
            else:
                review += line
    return data

def sentiment_twitter():
    return read_sentiment_data('twitter-tweets.txt')

def sentiment_yelp():
    return read_sentiment_data('yelp-reviews.txt')

def sentiment_imdb():
    return read_sentiment_data('imdb-reviews.txt')

def sentiment_amazon():
    return read_sentiment_data('amazon-reviews.txt')

def sentiment(bag_size=100):
    data_ = sentiment_yelp() + sentiment_imdb() + sentiment_amazon() + sentiment_twitter()

    # Calculate the bag_size most common words.
    all_words = Counter()
    for example, _ in data_:
        for word in example.split(' '):
            all_words[word] += 1
    bag_of_words = all_words.most_common(bag_size)

    # Create features (whether example has each of the words in bag).
    data = []
    labels = []
    word_in_example = lambda word, example: 1 if word in example else 0
    for example, label in data_:
        labels.append(int(label))
        data.append(np.array([word_in_example(word, example) for word, _ in bag_of_words]))

    # Shuffle data and separate into train and test set.
    data, labels = shuffle_data(data, labels)
    offset = int(len(data) * 0.7)
    return {
            'train': {
                'data': data[0:offset],
                'labels': labels[0:offset]
                },
            'test': {
                'data': data[offset:],
                'labels': labels[offset:]
                }
            }

def sentiment_nn(bag_size=100, offset=None):
    data_ = sentiment(bag_size)
    x_dim = len(data_['train']['data'][0])
    data = ClassificationDataSet(x_dim, 1)
    if offset:
        max_sample = offset
    else:
        max_sample = len(data_['train']['data'])
    for i in xrange(max_sample):
        data.addSample(data_['train']['data'][i], [data_['train']['labels'][i]])
    data_['train_nn'] = data
    return data_

