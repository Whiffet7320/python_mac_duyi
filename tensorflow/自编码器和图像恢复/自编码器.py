# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:12:14 2022

@author: Administrator
"""

import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist.load_data()
# 导入训练集的输入和标签   导入测试集的输入和标签
(train_fea, train_tar), (test_fea, test_tar) = mnist
train_fea,test_fea = tf.layers.flatten(train_fea),tf.layers.flatten(test_fea)
train_tar, test_tar = tf.one_hot(train_tar, 10), tf.one_hot(test_tar, 10)
val_fea, val_tar = train_fea[-5000:], train_tar[-5000:]
train_fea, train_tar = train_fea[:-5000], train_tar[:-5000]
# sess=tf.Session()
# img = tf.reshape(train_fea[0],shape=(28,28))
# noisy_img = img + 0.3 * np.random.randn(*img.shape)
# noisy_img = tf.clip_by_value(noisy_img,0.,1.)
# plt.imshow(noisy_img, cmap='Greys_r')

epochs = 80
batch_size = 128
encode_num = 100
lr = 0.1
noisy_rate = 0.3

image_size = train_fea.shape[1]

input_ = tf.placeholder(tf.float32, shape=[None, image_size], name='input_')
target_ = tf.placeholder(tf.float32, shape=[None, image_size], name='output_')

w1 = tf.Variable(tf.random_normal([image_size, encode_num], stddev=0.01))
b1 = tf.Variable(tf.zeros([encode_num ], tf.float32))
encode = tf.matmul(input_, w1) + b1
encode = tf.nn.relu(encode, name='encode')

w2 = tf.Variable(tf.random_normal([encode_num, image_size], stddev=0.01))
b2 = tf.Variable(tf.zeros([image_size ], tf.float32))
logits = tf.matmul(encode, w2) + b2  # logits要来算损失函数

decode = tf.nn.sigmoid(logits, name='decode')  # 经过sigmoid门才是最后输出的图像
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_, logits=logits)  # 算出一一对应的一组数
cost = tf.reduce_mean(loss)  # 算出一组数的平均loss

train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        # for i in range(train_fea.shape[0] // batch_size + 1):
        for i in range(10):
            new_train_fea, new_train_tar = train_fea[batch_size * epoch:batch_size * (epoch + 1)].eval(), train_tar[
                                                                                                          batch_size * epoch:batch_size * (
                                                                                                                  epoch + 1)].eval()
            # batch = mnist.train.next_batch(batch_size)
            noisy_batch = new_train_fea + noisy_rate * np.random.randn(*new_train_fea.shape)
            noisy_batch = np.clip(noisy_batch, 0., 1.)
            sess.run(train_op, feed_dict={input_: noisy_batch, target_: new_train_fea})
            # batch_loss = sess.run(cost,feed_dict={input_:batch[0],target_:batch[0]})
        batch_loss = sess.run(cost, feed_dict={input_: new_train_fea, target_: new_train_fea})
        print('epoch:{0:<3},batch_loss:{1:.3f}'.format(epoch, batch_loss))
    fig, axes = plt.subplots(2, 10, sharex=True, sharey=True, figsize=(20, 4))
    in_imgs = mnist.test.images[:10]
    noisy_in_imgs = in_imgs + noisy_rate * np.random.randn(*in_imgs.shape)
    noisy_in_imgs = np.clip(noisy_in_imgs, 0., 1.)
    decode_list = sess.run(decode, feed_dict={input_: noisy_in_imgs})

    for images, axes in zip([in_imgs, decode_list], axes):
        for img, ax in zip(images, axes):
            ax.imshow(img.reshape(28, 28), cmap='Greys_r')
