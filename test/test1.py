#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:32:56 2022

@author: yychen
"""


# 随机产生10个三位正整数的列表，然后按照他们个位数的大小进行排序。
import random;

def func(num):
    num = int(num)
    if num<100:
        return str(num + 100)
    else:
        return str(num)
    
ls = list(func(round(random.random(),3)*1000) for i in range(10))
print(ls)
newls = sorted(ls,key=lambda x:int(x[2]))
print(newls)



# 用至少两种方法实现将一个字符串倒序。
str1 = 'fisa';
ls2 = list(''.join(str1.split()))
ls2.reverse()
print(''.join(ls2))

str2 = 'hello'
print(str2[::-1])
