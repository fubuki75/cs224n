#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
'''
namedtuple https://www.cnblogs.com/linwenbin/p/11282492.html
* 可以产生一个可读性更强的tuple
* namedtuple函数返回的是一个tuple的子类对象，
* 不仅仅可以满足普通tuple中通过index来调用，还可以通过tuple的每个字段名称访问，
* 与普通tuple占用内存一致
'''

class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.encoder = None 
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None


        ### YOUR CODE HERE (~8 Lines)
        ### TODO - Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        #       W_h和W_c分别是encoder正逆向输出的拼接，在输入decoder前需要的映射
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        
        #LSTM模块里的dropout含义是在每个输出前加的
        self.encoder = nn.LSTM(embed_size,hidden_size,bidirectional=True)
        #在seq2seq的decoder中，要使用LSTM_cell，因为当预测到end-of-sentence时，要直接停止
        #讨论pytorch中LSTM 和 LSTM_cell的差异 https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell
        self.decoder = nn.LSTMCell(embed_size + hidden_size,hidden_size)#定义为LSTMCell的形式，由于参数共享，只用定义一个即可
        self.h_projection = nn.Linear(2*hidden_size, hidden_size,bias=False)#encoder是双向lstm，输出的合并，降维
        self.c_projection = nn.Linear(2*hidden_size, hidden_size,bias=False)#输出合并以及降维
        self.att_projection = nn.Linear(2*hidden_size, hidden_size,bias=False)
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size,bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, vocab.tgt.__len__(),bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)
        #我终于明白了，nn.Linear为啥有的时候bias=False了，因为 那个是为了用Linear替代matrix乘法

        ### END YOUR CODE

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.
        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.
        #encode，输出每个source位置词的hiddens（为attention做准备），以及encoder最终输出的hidden_state和cell_state
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        #生成蒙版,每个蒙板每个句子真实长度之外的为1，其余为0
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        #decoder输出所有的O_t，shape为(tgt_len,b,h)的综合结果，所有time_step都结束了
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        #分类问题，由decoder的输出得到概率P,shape(tgt_len,b,vocab_size)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        # Target_masks shape为(tgt_len, b)，选出非padded的部分为1，padded的部分为0
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        # torch.gather()函数主要用于提取tensor某个维度下特定index的信息 https://blog.csdn.net/cpluss/article/details/90260550 
        # torch.squeeze()在特定维度减少一维（仅当该维=1才有效），torch.unsqueeze()在特定位置加一维
        # target_padded维度为(tgt_len,b),index的维度为(tgt_len-1,b,1),去掉了<s>
        # target_gold_words_log_prob是盖上蒙板，shape为(tgt_len-1,b)
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]#乘以蒙板是排除掉空的
        #上面这一步其实是再算target_padded中ground truth对应的概率；
        # index=target_padded[1:].unsqueeze(-1)是每个batch每个ground truth词的概率
        #        (src_len-1,b)--unsqueeze-->(src_len-1,b,1)因为index的维度数和input要一致；
        # dim=-1是对最后一维vocab_size
        
        # scores长为b，每个句子的score
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.排过序
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
                                        每一步的hidden_state记录
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
                                                encoder的最后输出，也是decoder的初始化
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 8 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the decoder.
        ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        
        X = self.model_embeddings.source(source_padded)
        
        # pad_packed_sequence()函数的解释：避免pad输入到模型中引起误差 https://www.cnblogs.com/sbj123456789/p/9834018.html
        enc_hiddens, (last_hidden,last_cell) = self.encoder(pack_padded_sequence(X, source_lengths))
        #上面用到了pack_padded_sequence的话，仅enc_hiddens需要pad_packed_sequence.
        
        #pad_packed_sequence()返回seq_unpacked,lens_unpacked两个部分
        enc_hiddens = pad_packed_sequence(enc_hiddens)[0].permute(1,0,2)
        #enc_hiddens_1 = pad_packed_sequence(enc_hiddens,batch_first=True)[0]
        #上面两种写法等价，pad_packed_sequence的batch_first参数只用于控制输出的shape
        
        #判断两个tensor是否相等，不可==，而torch.equal()
        #assert torch.equal(enc_hiddens_2,enc_hiddens_1)
        
        # 以上：enc_hiddens (b,src_len,2*h) dec_init_state中h和c均为(2*num_layers,b,h),只不过num_layer默认取1
        # dec_init_state=(init_decoder_hidden, init_decoder_cell)
        # init_decoder_hidden的拼接-->(b,2*h)
        last_hidden = self.h_projection(torch.cat((last_hidden[0],last_hidden[1]),axis=1))
        last_cell = self.c_projection(torch.cat((last_cell[0],last_cell[1]),axis=1))
        dec_init_state = (last_hidden,last_cell)
        ### END YOUR CODE

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where 编码器每步输出，注意力需要
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where 蒙版
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder 解码器初始化
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        # seq2seq模型中encoder的输入是无句尾句首标志的，decoder的输入只有句首标记
        target_padded = target_padded[:-1]

        # 初始值Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # 初始值Initialize previous combined output vector o_{t-1} as zero
        # 第一个输入O_(t-1)不存在，设为0tensor，直接是一个batch的
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        # 先记录所有的o_t，最后一起算概率
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):按照dim维度分为size为多少的几块(第二个参数是每块的大小，不是块数！)，返回为tuple https://blog.csdn.net/qq_42518956/article/details/103882579
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack
        #attention score中h_decider*W*h_encoder，先计算后者W*h_encoder
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings.target(target_padded)
        #Y (tgt_len,b,e)
        for Y_t in torch.split(Y,1,dim=0):
            #squeeze Y_t to (b,e) , RNNCell无法接受三维
            Y_t = Y_t.squeeze(0)
            #组织decoder的输入，shape为(b,e+h)
            
            Ybar_t = torch.cat((Y_t,o_prev),dim=1)
            
            #step，利用LSTMcell以及dec_state（h，c），enc_hiddens和enc_hiddens_proj用于attention，enc_masks
            #输出dec_state(h,c是(b,h))
            dec_state, o_t, e_t = self.step(Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj,enc_masks)
            
            #最终的shape为(tgt_len,b,h)
            combined_outputs.append(o_t)
            o_prev = o_t
        #stack()函数的作用是将张量“序列”转换为一个完整的张量
        #与cat()函数的区别，cat在现有维度上加，stack是新加一个维度
        #[tgt_len个(b,h)张量]-->(tgt_len,b,h)
        combined_outputs = torch.stack(combined_outputs,dim=0)
        
        ### END YOUR CODE

        return combined_outputs


    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.
        
        怎么利用enc_masks

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """
        ### step，利用LSTMcell以及dec_state（h，c），enc_hiddens和enc_hiddens_proj用于attention，enc_masks
        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication: 这个方法非常好用，适用于带batch维度的计算：(b,n,m)*(b,m,p)=(b,n,p)
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        
        #hidden & cell shape (b,h)
        dec_state = self.decoder(Ybar_t,dec_state)
        dec_hidden,dec_cell = dec_state
        '''
        dec_hidden (b,h)--unsqueeze-->(b,h,1)
        enc_hiddens_proj (b,seq_len,h)
        e_t (b,seq_len)
        '''
        e_t = torch.bmm(enc_hiddens_proj,dec_hidden.unsqueeze(-1)).squeeze(-1)
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        # 这里开始操作mask
        if enc_masks is not None:
            # tensor.masked_fill(mask:bool tensor,带填充的值) 需要填充的是true
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:dim的限定，某个维度求和为1
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
        alpha_t = F.softmax(e_t, dim=-1)#一句话的概率之和为1 shape(b,seq_len)
        a_t = torch.bmm(alpha_t.unsqueeze(1),enc_hiddens).squeeze(1)

        #a_t (b,2h)
        U_t = torch.cat((a_t,dec_hidden),dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        ### END YOUR CODE
                

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.
        
        

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        #生成shape为(b,max source length)的0蒙版
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        #将每个句子真实长度之外的部分，蒙版设为1
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ 
        test时是一句一句进行翻译，这段代码相对比较复杂
        Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words) 只接收一句
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        # decoding的最大长度
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
        """
        #数据预处理：单句转化为单词list.  src_sents_var (src_len，1)
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        #编码：encode单个句子 src_encodings (1,src_len,h*2) dec_init_vec decoder的初始输入
        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        #注意力机制准备：W_attProj * h_enc 统一乘好了
        src_encodings_att_linear = self.att_projection(src_encodings)
        
        #初始化
        #decoder的初始状态
        h_tm1 = dec_init_vec
        #解码器输入的初始化，o_prev一开始肯定是0 [1,h]
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']
        #假设的初始化（目前只有一个，之后会expand）
        hypotheses = [['<s>']]
        #每个假设的分数(1,)
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        #完成的假设
        completed_hypotheses = []

        t = 0
        #大循环：一步步地解码，并在每一步结束时选择得分最大的当前序列，更新序列与得分等等
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:#停止条件：完成的hypotheses到达beam_size，或者encoding的时间步数达到最大
            t += 1
            hyp_num = len(hypotheses)#当前hypotheses的数量
            
            #####使用torch.expand()函数来将原来一句话的encodings变为多个，方便进行beam search
            
            #torch.expand()函数有点奇怪，对tensor某维度进行扩张：
            #如果是对老维度操作，则需要a原来的shape对应的老维度为1，a.shape (3,1)，只可a.expand(3,4)；若将维度设为-1则代表维度不改变
            #src_enccodings是最后所有的enc_hiddens，encoder中每一个单词的hiddens拼起来，(1,src_len,2*h)
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))#这步将shape变为(hyp与假设数一致,src_len,2*h)
            
            #exp_src_encodings_att_linear同理
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            
            #####将上一步topk个序列最新生成的单词，加上拼接o_t，进行一步解码
            
            #下一个时间片的输入
            #y_tm1 每个假设hyp上一步预测的词的嵌入(hyp,e)
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)
            
            #上一步的输出和att_tm1拼接，这里的att_tm1就是o_prev ;x(hyp,e+h)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)
            #利用step来顺序进行预测
            #这里enc_mask为None，是因为在source中不会有padding，一句一句读取的;att_t (hyp,h),h_t和cell_t也是(hyp,cell)
            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            #####解码完成后，计算每个词的概率，并计算不同序列的得分
            
            #o_t即att_t，softmax计算概率logP,shape(hyp,vocab_size)
            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            
            #live_hyp_num当前假设数量，有些已经预测到了eos_id，成为了candidate
            live_hyp_num = beam_size - len(completed_hypotheses)
            
            #beam内每个序列的持续得分：log P1P2P3P4...
            #原本为一维(hyp,)----->unsqueeze扩展为2维并expand为log_p_t的shape(hyp,vocab_size)
            # 这里相加是计算了之前hyp_score和下一步预测的单词对应所有可能性---->view为一维，混在一起，为了挑出live_hyp_num个最大的
            # continuating_hyp_scores (hyp*vocab_size,)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            
            #####选取前k个概率最大的，这里的操作是最巧妙的！！
            
            # 当前score中前k个概率最大的，返回相应的最大值与对应的index；top_cand_hyp_scores 和 top_cand_hyp_pos (live_hyp_num,)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            #确定是哪个假设的prev_hyp_ids，确定这个新预测出的单词是啥
            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt) #这里是不是错了
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)
            
            #####有了前k个概率最大的之后，要对序列、得分进行更新
            
            #下面的循环结束构，三者shape为(live_hyp_num,)
            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            
            #找出得分最高的live_hyp_num个的下一词，并记录下来
            #将假设id，单词位置以及得分循环，一个top一个top来，一共执行live_hyp_num次（当前beam里还有多少没确定下来）
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                #得到单个值
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                
                cand_new_hyp_score = cand_new_hyp_score.item()
                #得到 最大概率的词，并加到相应的假设上
                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':#如果预测的时候，预测完成了，将他放进value，去除首尾的标记，得分
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    #未结束，加到new假设中以便循环继续进行，
                    new_hypotheses.append(new_hyp_sent)
                    #目前的beam是延续的哪个假设的
                    live_hyp_ids.append(prev_hyp_id)
                    #记录最高得分
                    new_hyp_scores.append(cand_new_hyp_score)

            #####进行到这后，是否completed_hypotheses已经满了，终止；因为后续的操作可能会引起问题，等不到在大循环条件中判断了

            if len(completed_hypotheses) == beam_size:
                break
 
            #####根据选择的词，来更新decoder的状态和o_prev;保留假设和得分       
 
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            
            #选择相应的得分最大的当前序列，并记录下改序列下的hidden_state和cell_state(即h_tm1),以及o_prev情况
            #h_t和cell_t是(hyp,h)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
            
        #如果循环都结束了，还是没有完成的，说明太长了，则把hypotheses里第一个句子拿下（删除句首符号，句尾符号本身就不存在）
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
        #最后将生成的句子按照得分排序！
        #sort()函数 key和lambda的巧妙运用
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):#模型的加载
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        # NMT模型参数包括：词汇param['vocab'],
        # params['args']用于定义模型的参数（我一般是直接定义好的） 和 学习到的参数param['state_dict']
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)# **表示传入的是字典
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
