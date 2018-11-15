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

    def __init__(self,encode_dim=36,decode_dim=40,input_seq_len=121,output_seq_len=121,rnn_size=40, dnn_size = [64,128,64], layer_size=1,learning_rate=0.1,dropout=0.7,reg_lambda=0.3,train_X=None,train_Y=None,test_X=None,test_Y=None,sensor2sensor=True,train_epoch=5):
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.rnn_size = rnn_size
        self.layer_size = layer_size
        self.dnn_size = dnn_size
        self.learning_rate = learning_rate
        self.train_X = train_X # num_days * (num_recording_each_day * num_sample_each_recording) * dim_each_sample
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.sensor2sensor = sensor2sensor
        self.dropout = dropout
        self.reg_lambda = reg_lambda
        self.train_epoch = train_epoch
        self._get_simple_lstm(rnn_size,layer_size)
        self.buildFCModel()
        self.trainFCModel()

    def buildFCModel(self):
        self.input_seq = tf.placeholder(tf.float32, shape=[None, self.encode_dim], name='input_data')
        self.output_seq = tf.placeholder(tf.float32, shape=[None, self.decode_dim], name='output_data')
        self.weights = self.initialize_FCweights()
        self.linear_out = self.input_seq
        for i in range(len(self.dnn_size)):
            self.linear_out = tf.add(tf.matmul(self.linear_out, self.weights['layer_%d_W' %i]), self.weights['layer_%d_b' %i])
            self.linear_out = tf.nn.sigmoid(self.linear_out)
            if self.dropout:
                self.linear_out = tf.nn.dropout(self.linear_out, self.dropout)
        self.pred = tf.matmul(self.linear_out, self.weights['prediction_W'])
        # define regularization item by appling for loop:
        self.reg = 0
        for i in range(len(self.dnn_size)):
            self.reg += tf.contrib.layers.l2_regularizer(self.reg_lambda)(self.weights['layer_%d_W' %i])
        self.loss = tf.norm(tf.subtract(self.pred, self.output_seq)) + self.reg
        self.eval_loss = tf.norm(tf.subtract(self.pred, self.output_seq)) 
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize_FCweights(self):
        print('initializing weights ...')
        weights = {}
        glorot = np.sqrt(2/(self.encode_dim + self.dnn_size[0])) 
        weights['layer_0_W'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(self.encode_dim, self.dnn_size[0])), dtype=np.float32, name='FC_layer_0_weights')
        weights['layer_0_b'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1, self.dnn_size[0])),dtype=np.float32, name='FC_layer_0_bias')
        for i in range(1, len(self.dnn_size)):
            glorot = np.sqrt(self.dnn_size[1] + self.dnn_size[i-1])
            weights['layer_%d_W' %i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(self.dnn_size[i-1], self.dnn_size[i])), dtype=np.float32, name='FC_layer_%d_weights' %i)
            weights['layer_%d_b' %i] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1, self.dnn_size[i])), dtype=np.float32, name='FC_layer_%d_bias' %i)
        glorot = np.sqrt(2/(self.dnn_size[-1] + self.decode_dim))
        weights['prediction_W'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(self.dnn_size[-1],self.decode_dim)), dtype=np.float32, name='prediction_weights')
        return weights

    def trainFCModel(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for e in range(self.train_epoch):
            train_loss = 0
            for i in range(self.train_X.shape[0]):
                one_day_feature = self.train_X[i,:,:]
                one_day_label = self.train_Y[i,:,:]
                one_day_feature = one_day_feature.reshape(120*121,-1)
                one_day_label = one_day_label.reshape(120*121,-1)
                feed_dict = {
                    self.input_seq: one_day_feature,
                    self.output_seq: one_day_label
                }
                _, pred, label, loss, eval_loss = self.sess.run([self.train_op, self.pred, self.output_seq, self.loss, self.eval_loss], feed_dict=feed_dict)
                train_loss += loss / (121*120)
            print('training loss for epoch %d is:'%e, train_loss / self.train_X.shape[0])
            if e % 10 == 0:
                self.testFCModel(e)

    def testFCModel(self, e):
        test_loss = 0
        for i in range(self.test_X.shape[0]):
            one_day_feature = self.test_X[i,:,:]
            one_day_label = self.test_Y[i,:,:]
            one_day_feature = one_day_feature.reshape(120*121,-1)
            one_day_label = one_day_label.reshape(120*121,-1)
            feed_dict = {
                self.input_seq: one_day_feature,
                self.output_seq: one_day_label
            }
            _, pred, label, loss, eval_loss = self.sess.run([self.train_op, self.pred, self.output_seq, self.loss, self.eval_loss], feed_dict=feed_dict)
            test_loss += eval_loss / (121*120)
            ''' 
            for s in range(pred.shape[1]):
                plt.figure()
                plt.plot(pred[:,s])
                plt.plot(label[:,s])
                plt.savefig('./res/neural_plot/epoch_%d_' %e + 'day_%d' %i + '_sensor%d_res_autoencoder.png' %s) 
            '''
        print('testing loss is:', test_loss/self.test_X.shape[0])

    def buildLSTMModel(self):
        self.input_seq = tf.placeholder(tf.float32, shape=[None, self.input_seq_len, self.encode_dim], name='input_data')
        self.output_seq = tf.placeholder(tf.float32, shape=[None, self.output_seq_len, self.decode_dim], name='output_data')
        # encoder step to convert a seq to a vector-------------------------------------------
        encoder = self._get_simple_lstm(self.rnn_size, self.layer_size)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(encoder, self.input_seq, dtype=tf.float32)
        if self.sensor2sensor:
            #self.loss = tf.norm(tf.subtract(self.encoder_outputs,self.output_seq))
            self.loss = tf.losses.mean_squared_error(self.encoder_outputs, self.output_seq)
        else:
            # decoder step to decode the seq info-------------------------------------------------
            #decoder_cell = self._get_simple_lstm(self.rnn_size, self.layer_size)
            '''
            helper = TrainingHelper(self.output_seq, self.output_seq_len)
            decoder_cell = self._get_simple_lstm(self.rnn_size, self.layer_size)
            decoder = BasicDecoder(decoder_cell, helper, self.encoder_state)
            self.final_outputs, final_state, final_sequence_lengths = dynamic_decode(decoder)
            '''
            # loss function -----------------------------------------------------------------------
            #self.loss = tf.losses.mean_squared_error(self.final_outputs, self.output_seq) 
            #loss = tf.losses.mean_squared_error(logits_flat, self.output_seq)
        # training optimization function-------------------------------------------------------
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def trainLSTMModel(self):
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
                if i % 10 == 0:
                    plt.figure()
                    plt.plot(pred[0,:,0])
                    plt.plot(label[0,:,0])
                    plt.savefig('./res/neural_plot/epoch%d' %e + '_sensor%d_res.png' %i)
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


    def buildCNNModel(self):
        pass
        
    
