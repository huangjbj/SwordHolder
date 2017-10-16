import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

'''
将原始数据切割成 batch_size * N 的结构，ptb_producer实际是个queue，需要使用start_queue_runners启动才行
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
生成数据，x即为后续输入，y为预测结果，每次调用ptb_producer数据都会向前一步
[[4 3]
 [5 6]
 [1 0]] [[3 2]
 [6 1]
 [0 3]]

[[2 1]
 [1 1]
 [3 4]] [[1 0]
 [1 1]
 [4 1]]
'''