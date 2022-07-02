#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:53:07 2022

@author: yychen
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

rides = pd.read_csv('hour.csv')


dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for ele in dummy_fields:
    dummy = pd.get_dummies(rides[ele],prefix=ele)
    rides = pd.concat([rides,dummy],axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop,axis=1)

# 归一化
obj_mean_std={}
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
for ele in quant_features:
    mean,std = data[ele].mean(axis=0),data[ele].std(axis=0)
    # rides.loc[:,ele] = (rides[ele] - mean) / std
    data[ele] = (data[ele] - mean) / std
    obj_mean_std[ele] = [mean,std]

test_data = data[-21*24:]
data = data[:-21*24]
val_data = data[-60*24:]
train_data = data[:-60*24]


#将训练数据集数据拆分为 特征值 和目标值
target_fields = ['cnt', 'casual', 'registered']

test_fea,test_tar = test_data.drop(target_fields,axis=1),test_data[target_fields]
val_fea,val_tar = val_data.drop(target_fields,axis=1),val_data[target_fields]
train_fea,train_tar = train_data.drop(target_fields,axis=1),train_data[target_fields]

# 设置超参数
rl = 0.5
input_node = train_fea.shape[1]
hidden_node = 8
output_node = 1
step_num = 6000


fea = tf.placeholder(tf.float32,shape=(None,input_node))
tar = tf.placeholder(tf.float32,shape=(None,output_node))
w_i_to_h = tf.Variable(tf.random_normal((input_node,hidden_node),stddev=input_node**-0.5))
w_h_to_o = tf.Variable(tf.random_normal((hidden_node,output_node),stddev=hidden_node**-0.5))
h_input = tf.matmul(fea,w_i_to_h)
h_output = tf.sigmoid(h_input)
o_input = tf.matmul(h_output,w_h_to_o)
o_output = o_input
# sigmoid = lambda x:1/(1+np.exp(-x))

loss = tf.reduce_mean(tf.square(o_output - tar))
train = tf.train.GradientDescentOptimizer(rl).minimize(loss)
init = tf.global_variables_initializer()


# 这段代码是可以保存
def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
            # On the last batch, grab the rest of the data
        else:
            X, Y = x[ii:], y[ii:]
        # I love generators
        yield X, Y

obj1 = {'train':[],'val':[]}
with tf.Session() as sess:
    sess.run(init)
    for i in range(step_num):    
        batch = np.random.choice(train_fea.index, size=128)
        x,y = get_batches(train_fea,train_tar)
        # x,y = train_fea.iloc[batch].values,train_tar.iloc[batch]['cnt'].values[:,None]
        sess.run(train,feed_dict={fea:x,tar:y})
        # train(x, y, w_i_to_h, w_h_to_o)
        # train_loss = MSE(run(train_fea.values).T,train_tar['cnt'].values)
        train_loss = sess.run(loss,feed_dict={fea:train_fea.values,tar:train_tar['cnt'].values[:,None]})
        # val_loss = MSE(run(val_fea.values).T,val_tar['cnt'].values)
        val_loss = sess.run(loss,feed_dict={fea:val_fea.values,tar:val_tar['cnt'].values[:,None]})
        sys.stdout.write('\rProgress:{0:.1f},Train_loss:{1:.2f},Val_loss:{2:.2f}'.format(100*i/float(step_num),train_loss,val_loss))
        sys.stdout.flush()
        obj1['train'].append(train_loss)
        obj1['val'].append(val_loss)  
    new_test = sess.run(o_output,feed_dict={fea:test_fea.values})
    
# 画出两个loss的对比图
plt.figure()
plt.plot(obj1['train'], label='train_loss')
plt.plot(obj1['val'], label='val_loss')
plt.show()
plt.legend()
plt.show()

_,ax = plt.subplots(figsize=(8,4))
mean,std = obj_mean_std['cnt']
new_test = new_test * std + mean
ax.plot(new_test.reshape(new_test.shape[0],),label='test')
ax.plot((test_tar['cnt'] * std + mean).values,label='target')
ax.legend()
dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24],rotation=45)

    
















    






    
    
    
    
    
    
    