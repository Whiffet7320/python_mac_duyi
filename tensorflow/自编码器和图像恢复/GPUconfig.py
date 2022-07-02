# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:35:06 2022

@author: Administrator
"""

import tensorflow as tf
def getGPUconfig():
    gpu_device_name = tf.test.gpu_device_name()
    print(gpu_device_name)
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    config_photo = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)
    return config_photo

# with tf.device('/gpu:0'):
#     a = tf.constant('10',tf.string,name='a_const')
#     b = tf.string_to_number(a,out_type=tf.float64,name='str_2')
#     c = tf.to_double(5.0,name='to_double')
#     d = tf.add(b, c,name='add')
#     sess = tf.InteractiveSession(config=config_photo) #config就是前面配好的config_photo
#     print(d.eval())