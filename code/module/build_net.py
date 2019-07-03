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
from tensorflow.keras import backend
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

class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def transformer(n_symbols, embedding_weights, config):
    S_inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(input_dim=n_symbols,
                           output_dim=config.embeddingSize,
                           weights=[embedding_weights],
                           input_length=config.sequenceLength)(S_inputs)

    # 增加Position_Embedding能轻微提高准确率
    embeddings = Position_Embedding()(embeddings)
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(config.dropoutKeepProb)(O_seq)
    outputs = Dense(2, activation='softmax')(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def convolution(config):
    sequence_length = config.sequenceLength
    embedding_dimension = config.embeddingSize
    inn = Input(shape=(sequence_length, embedding_dimension, 1))
    cnns = []
    filter_sizes = config.filterSizes
    for size in filter_sizes:
        conv = Conv2D(filters=config.numFilters,
                      kernel_size=(size, embedding_dimension),
                      strides=1,
                      padding='valid',
                      activation='relu')(inn)
        pool = MaxPool2D(pool_size=(sequence_length - size + 1, 1),
                         padding='valid')(conv)
        cnns.append(pool)
    outt = concatenate(cnns)

    model = Model(inputs=inn, outputs=outt)
    return model


def cnn_mulfilter(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols,
                  output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),
        Reshape((config.sequenceLength,
                 config.embeddingSize,
                 1)),
        convolution(config),
        Flatten(),
        Dense(10,
              activation='relu',
              kernel_regularizer=regularizers.l2(config.l2RegLambda)),
        Dropout(config.dropoutKeepProb),
        Dense(1, activation='sigmoid')])

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


def lstm(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols,
                  output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),

        # LSTM层
        # LSTM(50,
        # activation='tanh',
        # dropout=0.5,
        # recurrent_dropout=0.5,
        # kernel_regularizer=regularizers.l2(config.model.l2RegLambda)),
        LSTM(50,
             activation='tanh',
             dropout=0.5,
             recurrent_dropout=0.5),

        Dropout(config.dropoutKeepProb),
        Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def gru(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),

        # GRU层
        # Bidirectional(GRU(100,activation='tanh', dropout=0.5, recurrent_dropout=0.5)),
        # TimeDistributed(Dense(50)),
        GRU(50, activation='tanh', dropout=0.5, recurrent_dropout=0.5),
        Dropout(config.dropoutKeepProb),
        Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = backend.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = backend.softmax(backend.tanh(backend.dot(x, self.W) + self.b))
        outputs = backend.permute_dimensions(a * x, (0, 2, 1))
        outputs = backend.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def lstm_attention(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),

        # LSTM层
        Dropout(config.dropoutKeepProb),
        LSTM(200, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
        # BatchNormalization(),
        AttentionLayer(),
        Dropout(config.dropoutKeepProb),
        Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def gru_attention(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),

        # LSTM层
        # LSTM(50,activation='tanh', dropout=0.5, recurrent_dropout=0.5,kernel_regularizer=regularizers.l2(config.model.l2RegLambda)),
        GRU(50, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
        Dropout(config.dropoutKeepProb),
        AttentionLayer(),
        Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def bi_lstm_attention(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),

        # LSTM层
        Dropout(config.dropoutKeepProb),
        Bidirectional(LSTM(50, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
                      merge_mode='concat'),
        AttentionLayer(),
        Dropout(config.dropoutKeepProb),
        # BatchNormalization(),
        Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def stack_lstm_attention(n_symbols, embedding_weights, config):
    model = Sequential([
        Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                  weights=[embedding_weights],
                  input_length=config.sequenceLength),

        # LSTM层
        Dropout(config.dropoutKeepProb),
        LSTM(50, activation='tanh', dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
        LSTM(50, dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
        LSTM(50, return_sequences=True),

        # BatchNormalization(),
        AttentionLayer(),
        Dropout(config.dropoutKeepProb),
        Dense(2, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model