#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#usage三句话的作用
    #训练
    #泛化？ 
    #检测    
"""
CS224N 2018-19: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    
    --log-every=<int>                       log every [default: 10]#每多少个循环输出一次日志
    --max-epoch=<int>                       max epoch [default: 30]#设置了默认值，太牛了，不写的话为None
    --input-feed                            use input feeding  #控制是否要将上一步的attention后的hidden状态和这一步的输入拼接
    
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5] #学习率衰减的写法
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000] #多少次循环之后（minibatch）来检验一下是不是该停止训练了
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time


from docopt import docopt #解析用户命令行参数
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union#用于给函数形参检测类型
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training#根据是否在训练返回bool型变量
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():#被with torch.no_grad()的代码不会track反向梯度，仍保持之前的梯度情况，适合于dev
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):#验证，就来一遍就行，还是一个batch一个batch来，不然可能装不下
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()#累计loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`#累计单词数
            cum_tgt_words += tgt_word_num_to_predict
        #去测试集上计算ppl，通过ppl来控制early stopping
        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    #每个句子的reference只有一句，一对一
    if references[0][0] == '<s>':#如果gold standard句子的首尾为<s>与</s>
        references = [ref[1:-1] for ref in references]
    #调用corpus_bleu函数
    bleu_score = corpus_bleu([[ref] for ref in references],#list[list[list[str]]]
                             [hyp.value for hyp in hypotheses])#list[list[str]]
    return bleu_score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    #把文件变为[[sent1],[sent2],...]
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    #读验证集
    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    #使用list(zip(a,b))能将a和b两者之间进行对齐；[([sent1_x],[sent2_y]),...]
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    #命令行参数读取，将字符转化为数字
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    
    vocab = Vocab.load(args['--vocab'])#vocab是第一步运行vocab.py即生成的.json文件
    #模型定义
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.train()
    
    uniform_init = float(args['--uniform-init'])#均匀分布，均值为0.1
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():#原来对于模型参数的初始化是这样的啊啊啊啊啊，一直没给模型初始化
            p.data.uniform_(-uniform_init, uniform_init)
    #vocab_mask？
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)
    #常规的adam优化器，看看如果在adam的基础上实现学习率的decay
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    #这里有意记录valid的分数
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):#结束该for循环就是一个epoch
            train_iter += 1 #循环数

            optimizer.zero_grad()

            batch_size = len(src_sents)
            #这里直接计算了loss，不需要外部定义，model的最终输出不是y，直接是log_P
            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()#sum()无任何参数：求总和
            loss = batch_loss / batch_size#别忘了除以batch_size
            #nn.MSELoss()建立出的损失函数，会自动对batch_size取平均；这里是因为是手动算的，需要除以batch_size

            loss.backward()

            # clip gradient  梯度截断，原来是在这里呀，在optimizer.step()之前，loss.backward()的后面
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            # loss为啥要加和，翻译的词数也要加和
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val#用于计算这log_every个batch内的平均loss，会按时清0
            cum_loss += batch_losses_val#cum代表累计损失

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>` target句没有句首<s>符号
            report_tgt_words += tgt_words_num_to_predict#记录词数，会按时清0，计算ppl用
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size#记录记句子数，会按时清0；计算平均损失用
            cum_examples += batch_size
            
            #每多少个循环输出一次日志
            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                #avg. loss展示每个句子的损失 report_loss / report_examples
                #ppl:perplexity 每个词的
                #cum. examples 累积词数
                #speed 每秒处理多少词
                train_time = time.time()#每次更新
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation 这里很重要，基于validation set判断是否该停止
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                #判断是不是最好的结果，考虑到了第一个
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
                
                #整体的逻辑：
                #若目前的最优，保存下来
                #结果不是最优时，累积一定次数（--patience）decay一次学习率且记录num_trial，decay一定次数后，再停止
                
                if is_better:
                    #目前最好的结果，保存模型，且patience清0（控制学习率衰减）
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    #！！！一定也要保存optimizer的状态
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                #当前结果不比之前所有的结果更好
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    
                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        #decay一定次数之后，就停止
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            #程序直接终止
                            exit(0)

                        # 衰减学习的写法！不仅要decay还要恢复到原来的model和优化器
                        # decay lr, and restore from previously best checkpoint
                        # 之前的定义里所有param的lr一致
                        
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        #这种参数加载方法，读model，但加载是加载model的参数
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        #这里map_location=lambda storage,loc:storage的含义是将GPU上的模型加载到CPU上；如果没有这些参数就是同种设备上的加载
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)
                        #load优化器
                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        #更新优化器的lr，需要循环，因为在定义优化器时，每个参数的初始lr可能不一样（本文一样）
                        #复习optimizer的定义,记得输入所有待优化的参数
                        #opt = torch.optim.SGD(model.parameters(),lr=0.01,**)#普通定义
                        #opt = torch.optim.Adam([
                        #                        {'params':model.parameters()},
                        #                        {'params':model.classifier.parameters(),'lr':0.0001}
                        #                       ]
                        #                       ,lr=0.01)
                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    负责test和泛化
    """
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:#是否有gold-standard句子，计算BLEU
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))
    #hypotheses的格式 List[List[Hypothesis]]
    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]#[0]代表取的是第一个hypothesis：得分最高的
        #top_hypotheses:list[Hypothesis ]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        #将翻译结果输出
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """
    #在程序的一开始（还没import）按照规则定义接口描述(usage,options)；然后命令行可被解析为参数字典，接下来根据这个字典编写逻辑
    # https://zhuanlan.zhihu.com/p/85748569 初步讲解大致用法
    # https://zhuanlan.zhihu.com/p/88083190 深入讲解usage的参数
    #注意：usage等一定要写在最开头（在import前），不然不可用；如果usage没写的功能，不可用
    args = docopt(__doc__)
    # Check pytorch version
    #assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
