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

# 想要debug跳转，需要 from .module.parameter_config import Config
from module.parameter_config import Config
from module.data_prepare import Dataset

from module.build_net import transformer
from module.build_net import cnn_mulfilter
from module.build_net import lstm
from module.build_net import gru
from module.build_net import lstm_attention
from module.build_net import gru_attention
from module.build_net import bi_lstm_attention
from module.build_net import stack_lstm_attention

if __name__ == "__main__":
    # 实例化配置参数对象
    config = Config()

    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # 生成词向量模型需要的数据
    with open('../data/goods_zh.txt', mode='r', encoding='utf-8') as f:
        sentences = []
        for line in f:
            temp = line.replace('\n', '').split(',,')
            sentences.append(jieba.lcut(temp[0]))

    # 生成词向量并保存词向量模型
    model = word2vec.Word2Vec(sentences,
                              size=110,
                              min_count=1,
                              window=10,
                              workers=multiprocessing.cpu_count(),
                              sg=1,
                              iter=20)
    model.save('../data/word2VecModel')

    # 加载保存好的词向量模型
    # model = gensim.models.Word2Vec.load('../data/word2VecModel')
    # model.wv.vocab.keys()
    # model['兢兢业业']  # 查看某个词的词向量

    data = Dataset(config)
    data.dataGen()

    print("train data shape: {}".format(data.trainReviews.shape))
    print("train label shape: {}".format(data.trainLabels.shape))
    print("eval data shape: {}".format(data.evalReviews.shape))
    print("eval label shape: {}".format(data.evalLabels.shape))

    wordEmbedding = data.wordEmbedding
    n_symbols = data.n_symbols

    ############################ 根据模型调整 ###########################
    model = transformer(n_symbols, wordEmbedding, config)
    # model = cnn_mulfilter(n_symbols, wordEmbedding, config)
    # model = lstm(n_symbols, wordEmbedding, config)
    # model = gru(n_symbols, wordEmbedding, config)
    # model = lstm_attention(n_symbols, wordEmbedding, config)
    # model = gru_attention(n_symbols, wordEmbedding, config)
    # model = bi_lstm_attention(n_symbols, wordEmbedding, config)
    # model = stack_lstm_attention(n_symbols, wordEmbedding, config)


    plot_model(
        model,
        to_file = 'img/model_transformer.png',
        show_shapes=True,
        show_layer_names=True
    )

    # plot_model(
    #     model,
    #     to_file='img/model_textcnn.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    # plot_model(
    #     model,
    #     to_file='img/model_lstm.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    # plot_model(
    #     model,
    #     to_file='img/model_gru.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    # plot_model(
    #     model,
    #     to_file='img/model_lstm_attention.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    # plot_model(
    #     model,
    #     to_file='img/model_gru_attention.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    # plot_model(
    #     model,
    #     to_file='img/model_bi_lstm_attention.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    # plot_model(
    #     model,
    #     to_file='img/model_stack_lstm_attention.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )

    model.summary()

    # 训练模型
    x_train = data.trainReviews
    y_train = data.trainLabels
    x_eval = data.evalReviews
    y_eval = data.evalLabels

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model_checkpoint = ModelCheckpoint('transformer_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
                                       save_best_only=True,
                                       save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('textcnn_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('lstm_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('gru_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('lstm_attention_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('gru_attention_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('bi_lstm_attention_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    # model_checkpoint = ModelCheckpoint('stack_lstm_attention_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5',
    #                                    save_best_only=True,
    #                                    save_weights_only=True)

    history = model.fit(x_train,
                        y_train,
                        batch_size=config.batchSize,
                        epochs=config.epochs,
                        validation_split=0.3,
                        shuffle=True,
                        callbacks=[reduce_lr, early_stopping, model_checkpoint])
    # 验证
    scores = model.evaluate(x_eval, y_eval)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

    # 保存模型
    yaml_string = model.to_yaml()
    with open('transformer_model/transformer.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('transformer_model/transformer.h5')

    # with open('textcnn_model/textCNN.yml', 'w') as outfile:
    #     outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    # model.save_weights('textcnn_model/textCNN.h5')

    # with open('lstm_model/lstm.yml', 'w') as outfile:
    #     outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    # model.save_weights('lstm_model/lstm.h5')

    # with open('gru_model/gru.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('gru_model/gru.h5')

    # with open('lstm_attention_model/lstm_attention.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('lstm_attention_model/lstm_attention.h5')

    # with open('gru_attention_model/gru_attention.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('gru_attention_model/gru_attention.h5')

    # with open('bi_lstm_attention_model/bi_lstm_attention.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('bi_lstm_attention_model/bi_lstm_attention.h5')

    # with open('stack_lstm_attention_model/stack_lstm_attenstion.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('stack_lstm_attention_model/stack_lstm_attention.h5')


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
    plt.savefig('img/loss_transformer.jpg')
    # plt.savefig('img/loss_textcnn.jpg')
    # plt.savefig('img/loss_lstm.jpg')
    # plt.savefig('img/loss_gru.jpg')
    # plt.savefig('img/loss_lstm_attention.jpg')
    # plt.savefig('img/loss_gru_attention.jpg')
    # plt.savefig('img/loss_bi_lstm_attention.jpg')
    # plt.savefig('img/loss_stack_lstm_attention.jpg')
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
    plt.savefig('img/accuracy_transformer.jpg')
    # plt.savefig('img/accuracy_textcnn.jpg')
    # plt.savefig('img/accuracy_lstm.jpg')
    # plt.savefig('img/accuracy_gru.jpg')
    # plt.savefig('img/accuracy_lstm_attention.jpg')
    # plt.savefig('img/accuracy_gru_attention.jpg')
    # plt.savefig('img/accuracy_bi_lstm_attention.jpg')
    # plt.savefig('img/accuracy_stack_lstm_attention.jpg')
    plt.show()