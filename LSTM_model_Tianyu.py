import tensorflow as tf
import numpy as np
import mpu.ml
from tensorflow.examples.tutorials.mnist import input_data
import os


def dataLoad(folder_path):
    pass
# this data is generated randomly
train_x = np.random.rand(300,50,10000)
train_y_idx = np.random.randint(1, 10, size = (10000,1))
train_y = np.array(mpu.ml.indices2one_hot([x[0] for x in train_y_idx], nb_classes=10))
print(train_y.shape)
# configuration 
lr = 0.001
num_epoch = 10
batch_size = 128

num_inputs = 50                # num of data input data
num_steps = 300                # time steps 
num_hidden_units = 128         # neurons in hidden layer
num_classes = 10               # error code classes (0-9 digits)


# input of the neural network
x = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_classes])

# Define weights
weights = {}
biases = {}
weights['in'] = tf.Variable(tf.random_normal([num_inputs, num_hidden_units]))     # (50, 128)
weights['out'] = tf.Variable(tf.random_normal([num_hidden_units, num_classes]))    # (128, 10)

biases['in'] = tf.Variable(tf.constant(0.1, shape=[num_hidden_units, ]))  # (128, )
biases['out'] = tf.Variable(tf.constant(0.1, shape=[num_classes, ]))           # (10, )

# def RNN function
def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X ==> (128 batch * 300 steps, 300 inputs)
    X = tf.reshape(X, [-1, num_inputs])
    # into hidden
    # X_in = (128 batch * 300 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 300 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, num_steps, num_hidden_units])

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    # hidden layer for output as the final results
    #results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results
# Build model  
out = RNN(x, weights, biases)
# Define loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
        #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # training part 
    for n in range(num_epoch):
        step = 0
        for i in range(train_x.shape[-1]//128 - 1):
            start = step * batch_size
            end = (step + 1) * batch_size
            batch_xs, batch_ys = train_x[:,:,start:end],train_y[start:end,:]
            batch_xs = batch_xs.reshape([batch_size, num_steps, num_inputs])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
                }))
            step += 1
