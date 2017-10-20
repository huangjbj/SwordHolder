# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:25:36 2017

@author: dell
"""

import tensorflow as tf
'''
暂时将stack和unstack视为作用于一维和二维数据上的函数
'''
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
c = tf.stack([a,b],axis=1)
c0 = tf.stack([a,b],axis=0)
d = tf.unstack(c,axis=0)
e = tf.unstack(c,axis=1)
print(c.get_shape())
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(c0))
    print(sess.run(d))
    print(sess.run(e))