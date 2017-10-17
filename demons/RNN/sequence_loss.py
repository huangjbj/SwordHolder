# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:35:15 2017

@author: dell
"""

import tensorflow as tf

with tf.Session() as sess:
    '''
    logits是4*3*2，因此batch_size = 4,sequence_length=3,num_decoder_symbols=2
    targets是4*3，
    num_decoder_symbols限定了targets中数的范围，也就是如果logits中词向量的长度为5，则targets中数的取值范围为[0,5)
    '''
    logits = tf.Variable([
            [
               [1,2,0],
               [3,4,0],
               [5,6,0]
            ],
            
            [
               [7,8,0],
               [9,10,0],
               [11,12,0]
            ],
            
            [
               [13,14,0],
               [15,16,0],
               [17,18,0]
            ],
            
            [
               [19,20,0],
               [21,22,0],
               [23,24,0]
            ]
       ], dtype = tf.float32)
    targets = tf.Variable([
            [0,1,2],
            [0,1,2],
            [0,1,2],
            [0,1,2]
            ])
    weights = tf.ones([4,3], dtype = tf.float32)
    sess.run(tf.global_variables_initializer())
    print(sess.run(weights))
    print(sess.run(tf.contrib.seq2seq.sequence_loss(logits, targets, weights)))