# 介绍

## 来源

使用ast分割python函数为若干语句，数据集是PCSD

## 作用

用于提取器判别是否是重要语句

## 数据组成

idx	数据索引号

cleaned_nl	去除字符的摘要

cleaned_codes	去除字符的代码

ex_labels	重要标签列表，0是不重要，1是重要，长度是分割的语句数量

fs	f1值

ps	准确率

rs	召回率

max_Rouge_l_r	最大rougeL分数

cleaned_seqs	分割好的语句

cleaned_seqs_ex	重要的语句
