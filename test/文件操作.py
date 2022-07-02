#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:50:44 2022

@author: yychen
"""

import jieba;
txt = open('gushi.txt',encoding='utf-8').read()
biaodian = '，。“”；‘’#¥%$（）*+.-/!！:：?？'
for tx in biaodian:
    if tx in txt:
        txt = txt.replace(tx,'')
words = jieba.lcut(txt)

counts = {}
for word in words:
    if len(word) == 1:
        continue
    else:
        counts[word] = counts.get(word,0)+1

items = list(counts.items())
items.sort(key=lambda x:x[1],reverse=True)
for i in range(10):
    word,count = items[i]
    print("{0}  {1}".format(word, count))