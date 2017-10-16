# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:29:50 2017

@author: Administrator
"""
from tensorflow.tutorials.rnn.ptb import reader
import tensorflow as tf
import numpy as np

DATA_PATH = 'simple-examples/data'
HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell, output_keep_prob = KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
        
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        if is_training: inputs = tf.nn.dropout(inputs, KEEP_PROB)
        
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN", reuse = False):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output , state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                
        #经过如此转换，output实际上是每个完整num_steps句子依次拼接
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        #output = tf.reshape(outputs, [-1, HIDDEN_SIZE])
        #print(output)
        
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        #print(logits)
        
        loss = tf.contrib.seq2seq.sequence_loss(
                )
        
        
with tf.variable_scope("test", reuse = True):        
    ptb = PTBModel(False,100,20)
'''
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:100])
'''


'''
raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
batch_size = 3
num_steps = 2
x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
with tf.Session() as session:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(session, coord=coord)
    try:
      xval, yval = session.run([x, y])
      print(xval, yval)
      xval, yval = session.run([x, y])
      print(xval, yval)

    finally:
      coord.request_stop()
      coord.join()
      '''

'''      
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

embedding = tf.Variable(np.identity(5, dtype=np.int32))
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(embedding.eval())
print(sess.run(input_embedding, feed_dict={input_ids:[1, 2, 3, 0, 3, 2, 1]}))
'''
'''
t1 = [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]  
t2 = [[10, 11, 12, 0], [13, 14, 15, 0], [16, 17, 18, 0]]  

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.concat([t1, t2], 0) ))
    print(sess.run(tf.concat([t1, t2], 1) ))
    '''
'''   
a = [
        [[1,2],
         [3,4],
         [5,6]
                ],
        [[7,8],
         [9,10],
         [11,12]
                ],
        [[13,14],
         [15,16],
         [17,18]
                ],
        [[19,20],
         [21,22],
         [23,24]
                
        ]
        ]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.reshape(tf.concat(a, 1), [-1, 2])))
    print(sess.run(tf.reshape(a, [-1, 2])))
    '''