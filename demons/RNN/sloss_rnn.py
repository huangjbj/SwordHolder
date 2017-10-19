# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:06:31 2017

@author: dell
"""

import tensorflow as tf

import numpy as np

'''
正态分布：均值0，标准差1，输出规模为[10,10]
'''
params=np.random.normal(loc=0.0,scale=1.0,size=[10,10])

encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[10,10])
decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[10,10])

logits=tf.placeholder(dtype=tf.float32,shape=[10,10,10])
targets=tf.placeholder(dtype=tf.int32,shape=[10,10])
weights=tf.placeholder(dtype=tf.float32,shape=[10,10])


train_encoder_inputs=np.ones(shape=[10,10],dtype=np.int32)
train_decoder_inputs=np.ones(shape=[10,10],dtype=np.int32)
train_weights=np.ones(shape=[10,10],dtype=np.float32)

num_encoder_symbols=10
num_decoder_symbols=10
embedding_size=10

cell=tf.nn.rnn_cell.BasicLSTMCell(10)

def seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size):
	encoder_inputs = tf.unstack(encoder_inputs, axis=0)
	decoder_inputs = tf.unstack(decoder_inputs, axis=0)
	results,states=tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    output_projection=None,
    feed_previous=False,
    dtype=None,
    scope=None
)
	return results

def get_loss(logits,targets,weights):
	loss=tf.contrib.seq2seq.sequence_loss(
		logits,
		targets=targets,
		weights=weights
	)
	return loss

results=seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size)
logits=tf.stack(results,axis=0)
print(logits)
loss=get_loss(logits,targets,weights)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	results_value=sess.run(results,feed_dict={encoder_inputs:train_encoder_inputs,decoder_inputs:train_decoder_inputs})
	print(type(results_value[0]))
	print(len(results_value))
	cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_decoder_inputs,
	                                 weights:train_weights,decoder_inputs:train_decoder_inputs})
	print(cost)