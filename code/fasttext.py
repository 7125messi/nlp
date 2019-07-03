import os
import time
import datetime
import random
import json
from collections import Counter
from math import sqrt
import gensim
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import logging
from gensim.models import word2vec
import multiprocessing
import yaml
import jieba

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Flatten

from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


# 超参数类
class Config(object):
    # 数据集路径
    dataSource = '../data/goods_zh.txt'
    stopWordSource = '../data/stopword.txt'

    # 分词后保留大于等于最低词频的词
    miniFreq = 1

    # 统一输入文本序列的定长，取了所有序列长度的均值。超出将被截断，不足则补0
    sequenceLength = 200
    batchSize = 64
    epochs = 10
    numClasses = 2

    # 训练集的比例
    rate = 0.8
    dropoutKeepProb = 0.5

    # 生成嵌入词向量的维度
    embeddingSize = 150

    ngram_range = 2
    max_features = 8000


# 实例化配置参数对象
config = Config()


# 数据预处理类，生成训练集和测试集
class Dataset(object):
    # 初始化参数
    def __init__(self, config):
        self.dataSource = config.dataSource
        self.stopWordSource = config.stopWordSource
        # 每条输入的序列处理为定长
        self.sequenceLength = config.sequenceLength
        self.embeddingSize = config.embeddingSize
        self.batchSize = config.batchSize
        self.rate = config.rate
        self.miniFreq = config.miniFreq

        self.stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None
        self.n_symbols = 0

        self.wordToIndex = {}
        self.indexToWord = {}

    # 读取数据集
    def readData(self, filePath):
        with open(filePath, mode='r', encoding='utf-8') as f:
            text = []
            label = []
            for line in f:
                temp = line.replace('\n', '').split(',,')
                text.append(temp[0])
                label.append(temp[1])

        print('data:the text number is {},the label number is {}'.format(len(text), len(label)))
        texts = [jieba.lcut(document.replace('\n', '')) for document in text]

        return texts, label

    # 读取停用词
    def readStopWord(self, stopWordPath):
        with open(stopWordPath, mode='r', encoding='utf-8') as f:
            stopWordList = f.read().splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    # 生成词向量和词汇-索引映射字典，可以用全数据集
    def genVocabulary(self, reviews):
        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]
        wordCount = Counter(subWords)  # 统计词频和排序
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= self.miniFreq]
        words.append('UNK')

        self.wordToIndex = dict(zip(words, list(range(len(words)))))
        self.indexToWord = dict(zip(list(range(len(words))), words))
        self.n_symbols = len(self.wordToIndex)
        config.max_features = self.n_symbols
        print('max_features:', self.n_symbols)

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open('../data/wordJson/wordToIndex.json', mode='w', encoding='utf-8') as f1:
            json.dump(self.wordToIndex, f1)

        with open('../data/wordJson/indexToWord.json', mode='w', encoding='utf-8') as f2:
            json.dump(self.indexToWord, f2)

    # 将数据集中的每条评论里面的词，根据词表，映射为index表示
    # 每条评论 用index组成的定长数组来表示
    def reviewProcess(self, review, sequenceLength, wordToIndex):
        reviewVec = [0] * sequenceLength
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]

        return reviewVec

    # 生成训练集和验证集
    def genTrainEvalData(self, x, y, rate):
        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self.reviewProcess(x[i], self.sequenceLength, self.wordToIndex)
            reviews.append(reviewVec)
            labels.append([y[i]])

        le = LabelEncoder()
        onehot_label = to_categorical(le.fit_transform(labels))
        x_train, x_eval, y_train, y_eval = train_test_split(reviews, onehot_label, test_size=rate)
        return x_train, y_train, x_eval, y_eval

    # 初始化训练集和验证集
    def dataGen(self):
        # 读取停用词
        self.readStopWord(self.stopWordSource)

        # 读取数据集
        reviews, labels = self.readData(self.dataSource)

        # 分词、去停用词
        # 生成 词汇-索引 映射表和预训练权重矩阵，并保存
        self.genVocabulary(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self.genTrainEvalData(reviews, labels, self.rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

# 构建模型
def transfromer(config):
    model = Sequential()
    model.add(Embedding(config.max_features, 200, input_length=config.sequenceLength))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(config.dropoutKeepProb))
    model.add(Dense(config.numClasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 构建 ngram 数据集
def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    从一个整数列表中提取  n-gram 集合。
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    增广输入列表中的每个序列，添加 n-gram 值
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

if __name__ == "__main__":
    data = Dataset(config)
    data.dataGen()

    x_train = data.trainReviews
    y_train = data.trainLabels
    x_test = data.evalReviews
    y_test = data.evalLabels

    nb_class = y_train.shape[1]
    n_symbols = data.n_symbols

    ngram_range = config.ngram_range
    max_features = config.max_features
    maxlen = config.sequenceLength

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.

        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer. 将 ngram token 映射到独立整数的词典
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        # 整数大小比 max_features 要大，按顺序排列，以避免与已存在的特征冲突

        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        # max_features 是可以在数据集中找到的最大的整数
        config.max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting X_train and X_test with n-grams features
        # 使用 n-gram 特征增广 X_train 和 X_test
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    # 填充序列至固定长度
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)

    # 建模并训练模型
    model = transfromer(config)
    plot_model(
        model,
        to_file='img/model_fasttext.png',
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  patience=10,
                                  mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)
    # 以每轮结束时checkpoint
    model_checkpoint = ModelCheckpoint('fasttext_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
                                       save_best_only=True,
                                       save_weights_only=True)
    history = model.fit(x_train,
                        y_train,
                        batch_size=config.batchSize,
                        epochs=config.epochs,
                        validation_split=0.3,
                        shuffle=True,
                        callbacks=[reduce_lr, early_stopping, model_checkpoint])

    # 验证
    scores = model.evaluate(x_test, y_test)

    # 保存模型
    yaml_string = model.to_yaml()
    with open('fasttext_model/fasttext_model.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('fasttext_model/fasttext_model.h5')
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))


    # 画图
    history_dict = history.history
    print(history_dict.keys())

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('img/loss_fasttext.jpg')
    plt.show()

    plt.clf()  # clear figure
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('img/accuracy_fasttext.jpg')
    plt.show()