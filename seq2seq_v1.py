import tensorflow as tf 
from tensorflow.contrib.seq2seq import *
import pickle
import numpy as np
from tqdm import tqdm
import time
import os
from os import listdir
from os.path import isfile, join
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import xgboost as xgb
from matplotlib import pyplot as plt
import pdb


class Seq2seqModel(object):

    def __init__(self,encode_dim=36,decode_dim=40,input_seq_len=121,output_seq_len=121,rnn_size=40,layer_size=1,learning_rate=0.1,train_X=None,train_Y=None,test_X=None,test_Y=None,sensor2sensor=True,train_epoch=5):
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.rnn_size = rnn_size
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.sensor2sensor = sensor2sensor
        self.train_epoch = train_epoch
        self._get_simple_lstm(rnn_size,layer_size)
        self.buildModel()
        self.trainModel()

    def buildModel(self):
        self.input_seq = tf.placeholder(tf.float32, shape=[None, self.input_seq_len, self.encode_dim], name='input_data')
        self.output_seq = tf.placeholder(tf.float32, shape=[None,self.output_seq_len, self.decode_dim], name='output_data')
        # encoder step to convert a seq to a vector-------------------------------------------
        encoder = self._get_simple_lstm(self.rnn_size, self.layer_size)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(encoder, self.input_seq, dtype=tf.float32)
        if self.sensor2sensor:
            #self.loss = tf.norm(tf.subtract(self.encoder_outputs,self.output_seq))
            self.loss = tf.losses.mean_squared_error(self.encoder_outputs, self.output_seq)
        else:
            # decoder step to decode the seq info-------------------------------------------------
            helper = TrainingHelper(self.output_seq, self.output_seq_len)
            decoder_cell = self._get_simple_lstm(self.rnn_size, self.layer_size)
            decoder = BasicDecoder(decoder_cell, helper, self.encoder_state)
            self.final_outputs, final_state, final_sequence_lengths = dynamic_decode(decoder)
            
            # loss function -----------------------------------------------------------------------
            self.loss = tf.losses.mean_squared_error(self.final_outputs, self.output_seq) 
            #loss = tf.losses.mean_squared_error(logits_flat, self.output_seq)
        # training optimization function-------------------------------------------------------
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def trainModel(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for e in range(self.train_epoch):
            train_loss = 0
            for i in range(self.train_X.shape[0]):
                one_day_feature = self.train_X[i,:,:]
                one_day_label = self.train_Y[i,:,:]
                one_day_feature = one_day_feature.reshape(120,121,-1)
                one_day_label = one_day_label.reshape(120,121,-1)
                feed_dict = {
                    self.input_seq: one_day_feature,
                    self.output_seq: one_day_label
                }
                _, pred, label, loss = self.sess.run([self.train_op, self.encoder_outputs, self.output_seq, self.loss], feed_dict=feed_dict)
                train_loss += loss / (121*120)
            print('training loss for epoch %d is:'%e, train_loss/self.train_X.shape[0])
            self.testModel()

    def testModel(self):
        test_loss = 0
        for i in range(self.test_X.shape[0]):
            one_day_feature = self.test_X[i,:,:]
            one_day_label = self.test_Y[i,:,:]
            one_day_feature = one_day_feature.reshape(120,121,-1)
            one_day_label = one_day_label.reshape(120,121,-1)
            feed_dict = {
                self.input_seq: one_day_feature,
                self.output_seq: one_day_label
            }
            _, pred, label, loss = self.sess.run([self.train_op, self.encoder_outputs, self.output_seq, self.loss], feed_dict=feed_dict)
            test_loss += loss / (121*120)          
        print('testing loss is:', test_loss/self.test_X.shape[0])

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layer = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layer)

    
