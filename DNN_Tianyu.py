import tensorflow as tf 
import numpy as np
import pdb
from matplotlib import pyplot as plt

class DNNModel():
    def __init__(self, train_X, train_Y, test_X, test_Y):
        self.encode_dim = 35
        self.decode_dim = 1
        self.dnn_size = [256, 512, 1024, 512, 256, 64, 1]
        self.batch_size = 128
        self.learning_rate = 0.001
        self.train_X = train_X # num_days * (num_recording_each_day * num_sample_each_recording) * dim_each_sample
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.reg_lambda = 0.002
        self.dropout = 0
        self.train_epoch = 500
        self.buildFCModel()
        self.trainFCModel()

    def buildFCModel(self):
        self.input_seq = tf.placeholder(tf.float32, shape=[None, self.encode_dim], name='input_data')
        self.output_seq = tf.placeholder(tf.float32, shape=[None, self.decode_dim], name='output_data')
        self.weights = self.initialize_FCweights()
        self.linear_out = self.input_seq
        for i in range(len(self.dnn_size)):
            self.linear_out = tf.add(tf.matmul(self.linear_out, self.weights['layer_%d_W' %i]), self.weights['layer_%d_b' %i])
            self.linear_out = tf.nn.tanh(self.linear_out)
            if self.dropout:
                self.linear_out = tf.nn.dropout(self.linear_out, self.dropout)
        self.pred = tf.matmul(self.linear_out, self.weights['prediction_W'])
        # define regularization item by appling for loop:
        self.reg = 0
        for i in range(len(self.dnn_size)):
            self.reg += tf.contrib.layers.l2_regularizer(self.reg_lambda)(self.weights['layer_%d_W' %i])
        self.loss = tf.losses.mean_squared_error(self.pred, self.output_seq) + self.reg
        self.eval_loss = tf.losses.mean_squared_error(self.pred, self.output_seq)
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
            start = 0
            end = self.batch_size
            for i in range(self.train_X.shape[0] // self.batch_size):
                one_day_feature = self.train_X[start : end, :]
                one_day_label = self.train_Y[start : end, :]
                feed_dict = {
                    self.input_seq: one_day_feature,
                    self.output_seq: one_day_label
                }
                _, pred, label, loss, eval_loss = self.sess.run([self.train_op, self.pred, self.output_seq, self.loss, self.eval_loss], feed_dict=feed_dict)
                train_loss += loss
                start += self.batch_size
                end += self.batch_size
            print('training loss for epoch %d is:'%e, train_loss / self.train_X.shape[0])
            if e % 10 == 0 and e != 0:
                print('----------------------------------')
                print('------------testing---------------')
                print('----------------------------------')
                self.testFCModel(e)


    def testFCModel(self, e):
        test_loss = 0
        start = 0
        all_pred = []
        all_label = []
        end = self.batch_size
        for i in range(self.test_X.shape[0] // self.batch_size):
            one_day_feature = self.train_X[start : end, :   ]
            one_day_label = self.train_Y[start : end, :]
            feed_dict = {
                self.input_seq: one_day_feature,
                self.output_seq: one_day_label
            }
            _, pred, label, loss, eval_loss = self.sess.run([self.train_op, self.pred, self.output_seq, self.loss, self.eval_loss], feed_dict=feed_dict)
            test_loss += eval_loss
            start += self.batch_size
            end += self.batch_size
            all_pred += list(pred)
            all_label += list(label)
        
        all_pred = np.array(all_pred)
        all_label = np.array(all_label)
        difference = (all_pred - all_label) / all_label * 100
        difference[np.isinf(difference)] = 0
        difference[np.isnan(difference)] = 0 
        plt.figure()
        plt.plot(all_pred[::30])
        plt.plot(all_label[::30])
        plt.savefig('./../res/neural_plot_twoline/epoch_%d_' %e) 
        
        print('testing loss is:', test_loss/self.test_X.shape[0])