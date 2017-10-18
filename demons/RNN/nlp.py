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
        
        self.cost = tf.contrib.seq2seq.sequence_loss(
                logits, self.targets, tf.ones([4,3], dtype = tf.float32)
                )
        self.final_state = state
        
        if not is_training: return
        
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
    
def run_epoch(session, model, data, train_op, output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    
    for step, (x, y) in enumerate(
            )
            
        
with tf.variable_scope("test", reuse = True):        
    ptb = PTBModel(False,100,20)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
