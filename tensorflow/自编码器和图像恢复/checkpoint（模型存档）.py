# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:36:26 2022

@author: Administrator
"""

import tensorflow as tf
from GPUconfig import getGPUconfig

tf.reset_default_graph()

# 存档
# 文件保存路径
# 命名为model.ckpt (.ckpt扩展名表示checkpoint)
save_file = './saveflie/model.ckpt'

w = tf.Variable(tf.truncated_normal(shape=(2,3)))
b = tf.Variable(tf.truncated_normal(shape=(3,)))
w1 = tf.Variable(tf.truncated_normal(shape=(2,3)))

saver = tf.train.Saver() #用来存取Tensor变量的类
with tf.Session(config=getGPUconfig()) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w),sess.run(b))
    saver.save(sess, save_file)

#移除之前的数据(w,b)
tf.reset_default_graph()
save_file = './saveflie/model.ckpt'
w1 = tf.Variable(tf.truncated_normal(shape=(2,3)))
b1 = tf.Variable(tf.truncated_normal(shape=(3,)))
saver = tf.train.Saver() #用来存取Tensor变量的类
with tf.Session(config=getGPUconfig()) as sess:
    saver.restore(sess, save_file) #加载之前存的Tensor变量
    print(sess.run(w1),sess.run(b1))
    
    
