# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:26:29 2017

@author: dell
"""

import tensorflow as tf

w1 = tf.Variable([[1.,2.]])
res1 = tf.matmul(w1, [[2.],[1.]])
grads1 = tf.gradients(res1, [w1])
grads2,_ = tf.clip_by_global_norm(grads1, 10)
grads3,_ = tf.clip_by_global_norm(grads1, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    '''
    grads1是直接计算出的梯度，为[2, 1]
    '''
    print(sess.run(grads1))
    '''
    grads2是截断的梯度，由于外部总梯度平方和的根为sqrt(5)<10,不需要截断，为[2, 1]
    '''
    print(sess.run(grads2))
    '''
    由于sqrt(5)>1，故对原梯度截断，为[2/sqrt(5), 1/sqrt(5)] = [0.8944, 0.4472]
    '''
    print(sess.run(grads3))