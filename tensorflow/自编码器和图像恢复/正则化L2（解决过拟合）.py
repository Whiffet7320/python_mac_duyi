# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:53:29 2022

@author: Administrator
"""


import tensorflow as tf
from GPUconfig import getGPUconfig

def get_w(shape,i):
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('noloss', w) #正则化权重
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(i)(w)) #正则化权重
    return w

if __name__ == '__main__':
    tf.reset_default_graph()
    fea = tf.placeholder(tf.float32,shape=(None,2))
    tar = tf.placeholder(tf.float32,shape=(None,1))
    batch_size = 8
    layer_net = [2,5,5,5,1] #定义五层神经网络，并设置每一层神经网络的节点数目
    i_input = fea
    i_layer_num = layer_net[0] #当前层的输入节点个数
    
    for i in range(1,len(layer_net)):
        o_layer_num = layer_net[i] #当前层输出节点个数
        w = get_w((i_layer_num, o_layer_num),0.001)
        b = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[o_layer_num,]))
        print(i_input.shape,w.shape)
        o_input = tf.nn.relu(tf.matmul(i_input, w) + b)
        i_layer_num = layer_net[i]
        i_input = o_input
    
    MSE_loss = tf.reduce_mean(tf.square(o_input - tar))
    tf.add_to_collection('loss', MSE_loss)
    loss = tf.reduce_sum(tf.get_collection('loss'))
    with tf.Session(config=getGPUconfig()) as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.get_collection('loss'),feed_dict={fea:[[2,3],[6,14]],tar:[[3],[14]]}))
        print(sess.run(tf.get_collection('noloss'),feed_dict={fea:[[2,3],[6,14]],tar:[[3],[14]]}))
        print(sess.run(loss,feed_dict={fea:[[2,3],[6,14]],tar:[[3],[14]]}))
    
    
    
    