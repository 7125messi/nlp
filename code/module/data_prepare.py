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
from tensorflow.keras.layers import GRU
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

class Dataset(object):
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

    def readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath, mode='r', encoding='utf-8') as f:
            stopWordList = f.read().splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        # 中文
        model = gensim.models.Word2Vec.load('../data/word2VecModel')
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("pad")
        wordEmbedding.append(np.zeros(self.embeddingSize))

        vocab.append("UNK")
        wordEmbedding.append(np.random.randn(self.embeddingSize))

        for word in words:
            try:
                # 中文
                vector = model[word]

                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + " : 不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(subWords)  # 统计词频，排序
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= self.miniFreq]

        # 获取词列表和顺序对应的预训练权重矩阵
        vocab, wordEmbedding = self.getWordEmbedding(words)

        self.wordEmbedding = wordEmbedding

        self.wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self.indexToWord = dict(zip(list(range(len(vocab))), vocab))
        self.n_symbols = len(self.wordToIndex) + 1

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open('../data/wordJson/wordToIndex.json', "w", encoding="utf-8") as f:
            json.dump(self.wordToIndex, f)

        with open('../data/wordJson/indexToWord.json', "w", encoding="utf-8") as f:
            json.dump(self.indexToWord, f)

    def reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论里面的词，根据词表，映射为index表示
        每条评论 用index组成的定长数组来表示

        """

        reviewVec = np.zeros((sequenceLength))
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

    def genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """

        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self.reviewProcess(x[i], self.sequenceLength, self.wordToIndex)
            reviews.append(reviewVec)

            labels.append([y[i]])

        trainIndex = int(len(x) * rate)

        trainReviews = sequence.pad_sequences(reviews[:trainIndex], maxlen=self.sequenceLength)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        trainLabels = to_categorical(trainLabels, num_classes=2)

        evalReviews = sequence.pad_sequences(reviews[trainIndex:], maxlen=self.sequenceLength)
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")
        evalLabels = to_categorical(evalLabels, num_classes=2)

        return trainReviews, trainLabels, evalReviews, evalLabels

    def dataGen(self):
        """
        初始化训练集和验证集
        """

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