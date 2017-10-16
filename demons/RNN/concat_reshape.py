# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:56:59 2017

@author: Administrator
"""
import tensorflow as tf

'''
如下是一个4*3*2的矩阵
'''
a = [
        [
           [1,2],
           [3,4],
           [5,6]
        ],
        
        [
           [7,8],
           [9,10],
           [11,12]
        ],
        
        [
           [13,14],
           [15,16],
           [17,18]
        ],
        
        [
           [19,20],
           [21,22],
           [23,24]
        ]
   ]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    '''
    两个reshape，都是将a转为12*2的矩阵
    但是第一个使用了concat进行第一步转化：可以看作将4个3*2矩阵在维度1拼接，得到3*（2*4）=3*8矩阵
    因此第一个输出为[1,2],[7,8],[13,14]...
    第二个输出为[1,2],[3,4],[5,6]...
    '''
    print(sess.run(tf.reshape(tf.concat(a, 1), [-1, 2])))
    print(sess.run(tf.reshape(a, [-1, 2])))