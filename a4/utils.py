#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    length = [len(x) for x in sents]
    max_len = max(length)
    sents_padded = [x+[pad_token]*(max_len-len(x)) for x in sents]
    ### END YOUR CODE

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.将file中每一句拆分[[sent1],[sent2]]
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence，仅对于y加入句首和句尾符号
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence；len(data)直接就是数据量
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)#math.ceil向上取整
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):#执行完一个for是一个epoch的所有句子
        #indices是一个尺寸为batch_size的list，里面是编号
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        #这样就完成了打乱，巧妙在打乱的是index，然后data[index]，example是一个句子中每个单词的list
        examples = [data[idx] for idx in indices]
        #只是每个batch内的句子从大到小进行排列，不是让模型从长句学到短句，因为在mini-batch内，顺序没有意义
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        #将x和y分开
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        #复习yield的用法
        #带有yield的函数python会把他当成一个generator，返回的是iterable对象，可以.next()的那种
        yield src_sents, tgt_sents

