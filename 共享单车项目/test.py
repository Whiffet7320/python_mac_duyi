#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:53:07 2022

@author: yychen
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
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
step_num = 2000

# 初始化权重
w_i_to_h = np.random.normal(0.0,input_node**-0.5,(input_node,hidden_node))
w_h_to_o = np.random.normal(0.0,hidden_node**-0.5,(hidden_node,output_node))
sigmoid = lambda x:1/(1+np.exp(-x))

# 该函数作用：给定输入特征值，通过正向传播输出预测结果 
def run(fea):
    # 正向传播
    h_input = np.matmul(fea,w_i_to_h)
    h_output = sigmoid(h_input)
    o_input = np.matmul(h_output,w_h_to_o)
    o_output = o_input
    return o_output

# 训练
def train(fea,tar,w_i_to_h,w_h_to_o):
    nr = fea.shape[0]
    # 正向传播
    h_input = np.matmul(fea,w_i_to_h)
    h_output = sigmoid(h_input)
    o_input = np.matmul(h_output,w_h_to_o)
    o_output = o_input
    # 反向传播
    e = o_output - tar[:,None] #误差值
    delta_o = e*1 #输出层残差
    #前一层残差 δ.W*f'(a)  h_output*(1-h_output) = f'(a)  
    delta_o_h = np.matmul(delta_o,w_h_to_o.T)*h_output*(1-h_output) 
    
    up_w_i_h_add = np.matmul(fea.T,delta_o_h) #输入层权重的和
    up_w_h_o_add = np.matmul(h_output.T,delta_o) #隐藏层权重的和
   
    #更新权重
    w_i_to_h -= rl*(up_w_i_h_add/nr)
    w_h_to_o -= rl*(up_w_h_o_add/nr) 
    
# 求标准差
def MSE(y,Y):
    return np.mean((y-Y)**2)
    

obj1 = {'train':[],'val':[]}
for i in range(step_num):    
    batch = np.random.choice(train_fea.index, size=128)
    x,y = train_fea.iloc[batch].values,train_tar.iloc[batch]['cnt'].values
    train(x, y, w_i_to_h, w_h_to_o)
    train_loss = MSE(run(train_fea.values).T,train_tar['cnt'].values)
    val_loss = MSE(run(val_fea.values).T,val_tar['cnt'].values)
    sys.stdout.write('\rProgress:{0:.1f},Train_loss:{1:.2f},Val_loss:{2:.2f}'.format(100*i/float(step_num),train_loss,val_loss))
    sys.stdout.flush()
    obj1['train'].append(train_loss)
    obj1['val'].append(val_loss)  
    
# 画出两个loss的对比图
plt.figure()
plt.plot(obj1['train'], label='train_loss')
plt.plot(obj1['val'], label='val_loss')
plt.legend()


_,ax = plt.subplots(figsize=(8,4))
mean,std = obj_mean_std['cnt']
new_test = run(test_fea) * std + mean
# ax.plot(new_test.values.reshape(15435,)[:5000],label='test')
# ax.plot((test_tar['cnt'] * std + mean).values.reshape(15435,)[:5000],label='target')
ax.plot(new_test.values.reshape(new_test.values.shape[0],),label='test')
ax.plot((test_tar['cnt'] * std + mean).values,label='target')
ax.legend()

dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24],rotation=45)

    
















    






    
    
    
    
    
    
    