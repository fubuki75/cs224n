# A4 NMT

## 代码架构分析

1. `utils.py`

   NMT任务的数据预处理的重要工具包，包含函数：

   1.1 `pad_sents`函数 将minibatch内的不等长的句子按照batch内最大长度使用pad_token补全

   1.2 `read_corpus`函数将数据（按行分隔\n）中的句子读取进来，拆分为`list[[str for sent1],[str for sent2],...]`，为tgt（也就是y）集添加句首句尾符号\<s>和\</s>

   1.3 `batch_iter`函数负责将整个数据集分成不同的minibatch（可选随机打乱），并在minibatch内的句子从长到短排序。

2. `model_embeddings.py`

   继承`nn.Module`建立`ModelEmbeddings`类，将提取出的词汇表vocab建立为nn.Embedding对象，并词嵌入是使用随机初始化。将会在包含vocab保存在NMT模型中，应该也可以在模型中直接定义。

3. `vocab.py`

   这个代码是单独使用的，直接用于从语料库中提取词汇表，内部定义了两个类：`VocabEntry`和`Vocab`.`VocabEntry`对象基于read_corpus后的结果转换为word-indice对应关系的dict，`Vocab`对象中存储这src和tgt两个`VocabEntity`对象。

   ```python
   #整个类的核心函数就是from_corpus，调用了__init__(),add函数，完成了从read_corpus后生成的corpus到word-indice字典的过程
   class VocabEntry(object):
       #构建word-indice的dict
       def __init__(self,word2id=None):
           #将word2id的字典保存至对象中，没有则建立几个常用元素（<pad>，句首句尾，<unk>），之后由add函数重新建立   
           #并建立id2word的字典
       def add(self,word):#将word2id和id2word中不包含的词加进去
       def from_corpus(corpus,size,freq_cutoff=2):#基于corpus构建VocabEntry对象，删除低频词,将corpus接入转为vocab_entry，生成词和id间的相互转换关系        
       #查询用的
       def __gititem__(self,word):#给出单词对应的id
       def id2word(self,wid):#给出self.idw2word中id对应的词
       def __contains__(self,word):#盖对象中的wordwid中是否含某个词
       def __len__(self):#一共多少次
       #预处理用的函数，将list[str]-->list[int]
       def words2indices(self,sent):#将句子转化为indices list[str]/list[list[str]]-->list[int]/list[list[int]]
       def indices2words(self,word_ids):#将indices再转化为句子
       def to_input_tensor(self, sents: List[List[str]], device: torch.device) ->torch.Tesnor#将句子转化为tensor，基于words2indices函数转换，利用pad_sents补全，最终转为tensor
   ```

   ```python
   class Vocab(object):
       def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):#将现成的src和tgt存进Vocab对象中
       def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':#调用vocab对象的from_corpus函数构建Vocab
       def save():
       def load():#这俩函数就是保存Vocab对象到本地罢了
   ```

4. `nmt_model.py`

   这个文件主要是实现了Luong在2015年的global注意力机制。* encoder和decoder的输入没有句尾符号，decoder的输出没有句首符号

   ```python
   class NMT(nn.Module):
       def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):#定义各种必要的层，将vocab存到ModelEmbedding对象里，再存储到model里，optimizer可直接在model.parameters()中优化到
       def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:#[仅在训练时使用]这个函数完成了所有的前向传播过程，输入是minibatch的raw data (List[List(str)])，输出是每个句子ground truth的对数概率（得分，便于计算损失）。其中调用了encode，decode，generate_sent_masks等函数
       def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:#利用LSTM模块实现encoder，输出为attention机制需要的每个词的hidden_state,和decoder的初始输入。
       def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:#主要是基于句子的真实长度，建立真实长度外的值为1，真实长度内的值为0的mask
       def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor, dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:#[只可在训练时使用，因为训练和测试对每个时间步的输入有差异]基于encoder的输出使用LSTMCell对每个时间步进行计算，每个时间步的输入为ground_truth对应的embedding和上一步注意力得到的o_t的拼接,并最终返回一个batch不分时间步的ouput
       def step(self, Ybar_t: torch.Tensor,dec_state: Tuple[torch.Tensor, torch.Tensor],enc_hiddens: torch.Tensor,enc_hiddens_proj: torch.Tensor,            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:#[训练和测试都会用到]带attention蒙版的一步运算
       def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:#[仅在测试时运行] 一句一句翻译，经过encode，attention，decode（没有调用decode函数，而是step函数）等多个步骤，返回所有可能性高的翻译结果的beam。
       def load(model_path: str):#模型的加载
   	def save(self, path: str):#模型的保存
   ```

   ### 补充：

   ① `forward`函数的思路：

   * __预处理__：计算minibatch内raw data的每个句子的长度`source_lengths`，调用`to_input_tensor`函数处理(pad补全、word2indice）minibatch内的raw data，生成`source_padded`和`target_padded`，但仅仅是list[list[int]]，还没有到词嵌入
   * __encode__：`self.encode(source_padded, source_lengths)`
   * __decode__：调用`generate_sent_masks`生成蒙版`enc_masks`（pad为0，真实有词为1），这里蒙版的意义是（见step函数）：用于attention机制，屏蔽$\mathbf{h}_{i}^{\text {enc }}$中每句话中的pad。decoder结果为__所有__time_step都结束了`self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)`
   * __输出__：对decode的结果进行projection（维度变换为vocab）后，进行log_softmax(方便计算损失)
   * __score__：基于tgt的ground truth生成target_masks（计算loss/score时只针对于ground truth的长度，预测值超出ground truth部分的loss/概率不管），计算每个句子真实长度下的score。

   ② `encode`函数思路:

   * __词嵌入__：调用`ModelEmbedding`类的对象，将句子的List[List[int]]转换为词嵌入。
   * __LSTM计算__：利用`pack_padded_sequence`来在pad补全的基础上在计算中避免引入pad带来的误差，进入encoder计算（LSTM），返回`enc_hiddens`和`last_hidden,last_cell`
   * __输出的转换__：`enc_hiddens`需要`pack_padded_sequence`，然后`last_hidden,last_cell`进行维度转换（矩阵乘法），作为`dec_init_state`。

   ③ `docode`函数思路：

   * __decoder初始化__：将encoder的hidden_state和cell_state传递；`o_prev`的初始零化（因为第一个o_prev不存在），一整个batch的

   * __attention计算__(之前一直没算)：$\mathbf{W}_{\text {attProj }} \mathbf{h}_{i}^{\text {enc }}$，但在现实实现中不分开算，直接用矩阵乘法算所有的，也方便soft_max

   * __预处理__：将target句子转为embedding，Y

   * __decode__（一个batch一个时间片一个时间片来）：

     ```python
     for Y_t in torch.split(Y,1,dim=0):#Y_t为一个batch每个时间片的tensor
         Y_t = Y_t.squeeze(dim=0)#LSTMCell只能接受2维
     ```

     拼接ground truth和`o_prev`,调用__`step`__函数（至关重要）来完成每一步的计算，并将每一步的`o_t`记下来（__这样之后的运算可以像encoder一样不分步__），更新`o_prev`。

   * __微处理__：将记下的`o_t`序列转为`tensor`

   ④ `step`函数思路

   * `LSTMCell`的decoder运算，得到`dec_hidden,dec_cell`

   * attention最终计算，生成`e_t`

   * 基于`enc_masks`与`e_t`生成真正的蒙版

     ```python
     if enc_masks is not None:
         e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
     ```

   * 对`e_t`进行`softmax`,与$\mathbf{h}_{i}^{\text {enc }}$相乘计算`a_t`

   * `U_t,V_t,O_t`的简单计算与返回

   ⑤ `beam_search`函数思路（仅在test时使用，不知道翻译后句子长度）

   * __预处理__：将单句转化为单词indice的列表

   * __encode__：完成encode，attention准备

   * __初始化__：初始化decoder的状态，以及hypotheses集合

     ```python
     #假设的初始化（目前只有一个，之后会expand）
     hypotheses = [['<s>']]
     #每个假设的分数(1,)
     hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
     #完成的假设
     completed_hypotheses = []
     ```

   * __decode和beam search__：使用大循环，一步一步解码，并在每步结束时选择当前得分前beam_size个序列，更新序列和得分。

     终止条件 :

     直到beam_size个序列都预测到了句尾符`len(completed_hypotheses) < beam_size`

     或者到达了最大时间步` t < max_decoding_time_step`

     ```python
     while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
         t += 1
         hyp_num = len(hypotheses)#获取当前hypotheses的数量
         
         #####1.使用torch.expand()函数来将原来一句话的encodings复制为多个（并和在一个张量里），方便进行beam search，attention准备同理  
         #src_enccodings是最后所有的enc_hiddens，这步将encodings复制hyp次，变为(hyp与假设数一致,src_len,2*h)
         exp_src_encodings = src_encodings.expand(hyp_num,
                                                  src_encodings.size(1),
                                                  src_encodings.size(2))
         #attention准备进行类似的操作
         exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,  
                                                                        src_encodings_att_linear.size(1),
                                                                        src_encodings_att_linear.size(2))
     
         #####2.从hypotheses中取出上一步topk个序列最新生成的单词，加上拼接o_t，作为这个时间片的输入，进行一步解码
         #y_tm1 每个假设hyp上一步预测的词的嵌入(hyp,e)
         y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
         y_t_embed = self.model_embeddings.target(y_tm1)
         #上一步的输出和att_tm1拼接，这里的att_tm1就是o_prev ;x(hyp,e+h)
         x = torch.cat([y_t_embed, att_tm1], dim=-1)
         #利用step函数来进行一步预测
         #这里enc_mask为None，是因为在source中不会有padding，一句一句读取的;att_t (hyp,h),h_t和cell_t也是(hyp,cell)
         (h_t, cell_t), att_t, _  = self.step(x, h_tm1,exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)
     
         #####3.解码完成后，计算每个词的概率，并计算不同序列的得分
         #o_t即att_t，softmax计算概率logP,shape(hyp,vocab_size)
         log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
         #live_hyp_num当前假设数量，有些已经预测到了eos_id，成为了candidate
         live_hyp_num = beam_size - len(completed_hypotheses)
         
         #计算不同序列的得分，这里很精彩！！也是通过expand
         #beam内每个序列的持续得分：log P1P2P3P4...
         #原本为一维(hyp,)----->unsqueeze扩展为2维并expand为log_p_t的shape(hyp,vocab_size)
         # 这里相加是计算了之前hyp_score和下一步预测的单词对应所有可能性---->view为一维，混在一起，为了后面挑出live_hyp_num个最大的
         # continuating_hyp_scores (hyp*vocab_size,)
         contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
     
         #####4.选取前k个概率最大的，这里的操作是最巧妙的！！
         top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)# 当前score中前k个概率最大的
         prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)#确定源于哪个假设
         hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)#确定这个新预测出的单词是啥
     
         #####5.有了前k个概率最大的之后，要对序列、得分进行更新
         #下面的循环结束时，三者shape为(live_hyp_num,)
         new_hypotheses = []
         live_hyp_ids = []#用于锁定下一步采用哪个h_t,cell_t
         new_hyp_scores = []
         #找出得分最高的live_hyp_num个的下一词，并记录下来
         #将topk对应的 假设，单词以及得分 循环，一个一个来，一共执行live_hyp_num次（当前beam里还有多少没确定下来）
         for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
             prev_hyp_id = prev_hyp_id.item()#tensor.itm()得到tensor（必须是只有一个元素的）的数值
             hyp_word_id = hyp_word_id.item()
             cand_new_hyp_score = cand_new_hyp_score.item()
             #得到 最大概率的词，并加到相应的新假设上，必须得是新的，因为topk可能出自于同一个的序列
             hyp_word = self.vocab.tgt.id2word[hyp_word_id]
             new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
             if hyp_word == '</s>':#如果预测的时候，预测完成了，将他放进value，去除首尾的标记，得分
                 completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],score=cand_new_hyp_score))
             else:
                 new_hypotheses.append(new_hyp_sent)#往新假设里加
                 live_hyp_ids.append(prev_hyp_id)#目前的beam是延续的哪个假设的
                 new_hyp_scores.append(cand_new_hyp_score)#记录最高得分      
         
         #####6.进行到这后，是否completed_hypotheses已经满了，终止；因为后续的操作可能会引起问题，等不到在大循环条件中判断了
         if len(completed_hypotheses) == beam_size:
             break
     
         #####7.根据选择的词，来更新decoder的状态和o_prev;保留假设和得分       
         live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
         #选择相应的得分最大的当前序列，并记录下改序列下的hidden_state和cell_state(即h_tm1),以及o_prev情况
         h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
         att_tm1 = att_t[live_hyp_ids]
     	hypotheses = new_hypotheses
      	hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
     ```

   * __特殊情况（鲁棒性）__：如果循环都结束了，还是没有完成的假设，说明太长了，则把hypotheses里第一个句子拿下（删除句首符号，句尾符号本身就没预测到）

   * __排序__：按照每句的得分（log概率连加）排序

5. `run.py`

   文件通过命令行中的子命令控制了具体的运行：

   * 训练(`run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]`)
   * 检测（`run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE`）
   * 泛化（`run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE`）

   ```python
   def train(args:Dict):#模型训练与输出，lr decay和early stopping
   def evaluate_ppl(model, dev_data, batch_size=32):#基于目前的model和dev_data计算当前的perplexity
   def decode(args: Dict[str, str]):#负责检测与泛化（区别在于是否提供ground truth）
   def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:#仅针对每句话就一个reference的情况，计算corpus的bleu得分，调用ntlk.translate.bleu_score库中的corpus_bleu函数计算
   def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:#循环所有的句子，利用nmt_model.py中的beam_search函数来一句一句翻译
   ```
   
   ### 补充：
   
   ① `train`函数思路：
   
   * 读取命令行`args`中的一切参数，模型定义与__初始化__
   
   * 分别读取训练集和验证集的src和tgt，通过`list(zip(src,tgt))`将两者进行对齐，并放入一个列表
   
   * 定义优化器
   
   * 嵌套循环训练：终止条件是达到最大epoch数或early stopping
   
     * `batch_iter`函数作为minibatch生成器，由于`shuffle=True`每个epoch的minibatch不同，__更科学__
   
     * 常规方法预测，求每个样本（句子）平均loss（如果使用nn内部的损失函数直接得到的平均损失，自己写损失要注意），梯度计算，__clip gradient__，更新一步
   
     * 记录loss
   
       基于直接获得的`batch_loss`或者是`avg_batch_loss`，计算平均句子/单词loss等
   
     * 每隔`log_every`循环输出一次训练信息
   
       ```python
       print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '\'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' 
             % (epoch,train_iter, #epoch，iter信息
                report_loss / report_examples, #连续几个iter的每个样本平均损失
                math.exp(report_loss / report_tgt_words),#平均每个词的perplexity
                cum_examples, #累计的句子数
                report_tgt_words / (time.time() - train_time), #速度
                time.time() - begin_time), file=sys.stderr)#时间
       ```
   
     * 每隔`valid_niter`个循环进行验证，根据验证结果，进行lr_decay以及early_stopping。采用的评价指标是：每个词的负的perplexity。运用lr_decay和early_stopping的规则为：
   
       ① 若目前的最优，保存下来模型和优化器的状态,patience归零；
   
       ② 结果不是最优时，__连续__出现一定次数（--patience），decay一次学习率且记录num_trial；num_trail__累积__到达一定次数后（decay了一定次数后），early stopping。
     
     ② `decode`函数思路
     
     * 读取数据
     * 运行`beam_search`函数
     * （测试的话，）计算`bleu_score`
     * 翻译结果输出(得分最大的句翻译)：利用`' '.join(top_hyp)`将单词list形成句子
   

# 收获

## 其他收获

1. `from typing import List, Tuple, Dict,Union`

   这个主要用于给形参一个类型检测，在形参后面`:形参类型`，`()->返回值类型`，当不满足时会警告，但不会出错。但是对于Tensor无法确定其具体的shape。

   ```python
   def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
   ```

2. 一行行地读取文本文件的内容

   ```python
   for line in open(file_path):
       sent = line.strip().split(' ')
       #strip([chars])函数是删除字符串 首尾 的字符，chars是字符集，没有顺序可言
       #'123421'.strip('12')  output为'34'
       #split(str="")函数是指定分隔标志将字符串切片
   ```

3. 完整数据集分成minibatch的generator的手动实现 （打乱顺序的实现，generator的生成：for循环内的yield）

   ```python
   def batch_iter(data, batch_size, shuffle=False):
       batch_num = math.ceil(len(data) / batch_size)#math.ceil向上取整
       #打乱数据顺序的手动实现：打乱index，再重新按照index装填
       index_array = list(range(len(data)))
       if shuffle:
           np.random.shuffle(index_array)
         
       for i in range(batch_num):#执行完一个for是一个epoch的所有句子
           indices = index_array[i * batch_size: (i + 1) * batch_size]#不足长度的索引依旧可以进行，对于list，array，tensor都可以。
           examples = [data[idx] for idx in indices]
           examples = sorted(examples, key=lambda e: len(e[0]), reverse=True) #排序
           #将x和y分开
           src_sents = [e[0] for e in examples]
           tgt_sents = [e[1] for e in examples]
           #带有yield的函数python会把他当成一个generator，返回的是iterable对象，可以.next()的那种
           yield src_sents, tgt_sents
   ```

4. `sorted`函数中`key`参数与`lambda`的巧妙运用

   ```python
   sorted(a,key=lambda x:len(x))
   ```

5. 如果字典的key和value是一一对应时，且需要将该字典key-value调换，则可使用：

   ```python
   id2word = {v:k for k,v in word2id.items()}
   ```

6. 在if语句中判断是否为某类

   ```python
   if type(a)==list:#可以这么写
   ```

7. 基于列表生成式，生成多维度列表

   ```python
   [[self[w] for w in s] for s in sents]#sents
   ```


8. `from itertools import chain` chain工具：将iterable相连为更大的

   ```python
   Counter(chain(*corpus))#corpus为list[list[str]]，经过chain(*)后变成一维
   ```

9. `*list *tuple **dict`作为形参 (https://www.cnblogs.com/hider/p/14707088.html)

   ```python
   *list和*tuple是将每个元素分开按顺序作为形参
   **dict则必须key与形参一模一样才可
   ```

10. `from docopt import docopt`解析用户命令行的工具

    在程序的一开始（`import`前，否则不可用）按照规则定义接口描述(usage,options)；简单通过下列命令，命令行可被解析为参数字典`args`，接下来根据这个字典编写逻辑，比如`args['--cuda']`

    ```python
    args = docopt(__doc__)
    ```

    * https://zhuanlan.zhihu.com/p/85748569 初步讲解大致用法
    * https://zhuanlan.zhihu.com/p/88083190 深入讲解usage的参数
    * 注意：在usage中没写的命令行格式，不可用。


## pytorch相关

1. 模型训练的种种细节

   * 模型创建后的各层参数的初始化：

     ```python
     #均匀分布
     for p in model.parameters():
         p.data.uniform_(-uniform_init, uniform_init)
     #xavier分布
     for p in model.parameters():
             if p.dim() > 1:
                 nn.init.xavier_uniform(p)
     ```

   * `model.parameters()`(多用于优化器初始化)中包含了这个Module中所有可学习的参数（不管是不是`requires_grad=True`），包括各种层以及`nn.Parameter()`，但单纯的tensor及时`requires_grad=True`也不在里面（https://www.jianshu.com/p/d8b77cc02410）;model.state_dict()（多用于保存参数）装的参数与其类似，可学习的参数。

   * Variable, Parameter之间的差异 （基本上不用Variable了） https://blog.csdn.net/u014244487/article/details/104372441

   * clip gradient的实现

     ```python
     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)#clip_grad的是梯度截断的最大梯度值
     ```

   * validation, lr_decay, early stopping的写法

     ```python
     cum_loss = cum_examples = cum_tgt_words = 0.
     valid_num += 1
     #begin validation
      # compute dev. ppl and bleu
     dev_ppl = evaluate_ppl(model, dev_data, batch_size=128) 
     valid_metric = -dev_ppl
     
     is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)#判断是否是目前所有次验证中最好的结果
     hist_valid_scores.append(valid_metric)#记录下这次的metric
     if is_better:
         patience = 0
         model.save(model_save_path)
         torch.save(optimizer.state_dict(), model_save_path + '.optim')
     elif patience < int(args['--patience']):    #当前结果不比之前所有的结果更好
         patience += 1
         if patience == int(args['--patience']):
            num_trial += 1
            #decay一定次数之后，就停止
            if num_trial == int(args['--max-num-trial']):
               print('early stop!', file=sys.stderr)
     	      exit(0)#程序直接终止
             
            # 衰减学习的写法！不仅要decay，还要恢复到原来的model和优化器！
     	  lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])# 获取之前的lr，乘以decay_rate得到lr
            #load model
            params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            model = model.to(device)
            #load优化器
            optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
             # set new lr
             for param_group in optimizer.param_groups:
                 param_group['lr'] = lr
             # reset patience
             patience = 0
      #early stopping的特殊情况
      if epoch == int(args['--max-epoch']):
     	print('reached maximum number of epochs!', file=sys.stderr)
     	exit(0)
     ```

   * 验证集怎么算loss(`with torch.no_grad(), model.eval()`等命令)

     ```python
     was_training = model.training#根据是否在训练返回bool型变量
     model.eval()
     with torch.no_grad():#被with torch.no_grad()的代码不会track反向梯度，仍保持之前的梯度情况，适合于dev
         pass
     if was_training:
        model.train()
     ```

1. LSTM相关

   * 在seq2seq中，encoder需要LSTM，decoder需要使用LSTMCell。因为，在decoding时不明确翻译后的长度（不知道真值），预测到</s>即停止，方便进行控制。[帖子](https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell)讨论了两者之间的差异。*注：其中LSTMCell的输出仅有hidden_state，没有output。
   * 由于参数共享，LSTMCell就定义一个即可
   * LSTMCell的输入只可是2维的

2. 注意力相关

   * 注意力计算中第一步是score的计算$\mathbf{e}_{t, i}=\left(\mathbf{h}_{t}^{\text {dec }}\right)^{T} \mathbf{W}_{\text {attProj }} \mathbf{h}_{i}^{\text {enc }}$,其中$\mathbf{W}_{\text {attProj }}$由`nn.Linear(,bias=0)`来实现

   * LSTM输出的ouput即为每个source位置词的hiddens( $\mathbf{h}_{i}^{\text {enc }}$)，为score计算中$\mathbf{h}_{i}^{\text {enc }}$

   * __attention计算__：$\mathbf{W}_{\text {attProj }} \mathbf{h}_{i}^{\text {enc }}$，但在现实实现中不分开算$\mathbf{W}_{\text {attProj }} \mathbf{h}^{\text {enc }}$，直接用矩阵乘法算所有的，也方便soft_max

   * 最后一步$\left(\mathbf{h}_{t}^{\text {dec }}\right)^{T} \mathbf{W}_{\text {attProj }} \mathbf{h}_{i}^{\text {enc }}$，通常使用`torch.bmm`完成

     ```python
     e_t = torch.bmm(enc_hiddens_proj,dec_hidden.unsqueeze(-1)).squeeze(-1)
     ```

     而且计算后的`e_t`需要经过$ \mathbf{h}_{i}^{\text {enc } }$的mask(5.蒙版 #2)

3. seq2seq Luong的global attention中一共用到两个mask

   * `enc_masks`：用于attention机制，屏蔽$\mathbf{h}_{i}^{\text {enc }}$中每句话中的pad

   * `target_masks`：用于训练时，在给出每个翻译后句子score（损失）时，不计算原句长度之外的score，__因为在decode函数中，Y直接传入，没有筛去pad，自然也会自动decode中输入pad，预测__。

     __注意__：模型__训练__中，encode中的输入通过`pack_padded_sequence`处理了pad问题，decode的attention通过`enc_masks`处理了pad问题，decode的输入__没有__处理pad的问题，会出现输入pad预测的现象，但在计算损失的时候只管ground truth长度下对应的损失。

4. `nn.Linear()`巧用

   * `nn.linear`层等价于矩阵乘法，但对多于两维的张量计算时，仅针对后两维计算：

     ```python
     #a (tgt_len,b,h)
     #linear=nn.Linear(h,vocab_size)
     self.linear(a)#shape为(tgt_len,b,vocab_size)
     ```

   * 一个bias=0的情况：公式推导中的矩阵间乘法，可以用`nn.Linear`，其中`bias=0`来等价实现

5. 蒙版

   蒙版生成：可以利用`broadcast`性质

   ```python
   #1
   for e_id, src_len in enumerate(source_lengths):
       enc_masks[e_id, src_len:] = 1
       
   #2 利用tensor.data.mask_fill函数
   #tensor.masked_fill(与tensor shape相同的mask:bool tensor,待填充的值) bool tensor需要填充的是true
   e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
   
   #3 target_padded有维度，=右侧为int值，结果为bool张量，.float()转换
   target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
   ```

   蒙版使用（*）

   ```python
   #直接相乘即可（elementize multiplication），前提是两种shape一致
   a*target_masks
   ```

6. `log_softmax`函数

   `dim`参数代表沿哪一维加和为1

   ```python
   F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
   ```

7. [`torch.gather()`]( https://blog.csdn.net/cpluss/article/details/90260550 )

   主要用于提取tensor某个维度下特定index的信息，用于从`(tgt_len,b,vocab_size)的张量`中取出ground truth每个单词的概率`(tgt_len-1,b,1)` (-1是删去\<s>句首符)

   * 参数tensor
   * 参数index的维度数应该与P一致，最终的输出与index的shape一致，基于index构建输出
   * 参数dim，所针对的那个维度进行提取

   ```python
   # P维度为(tgt_len,b,vocab_size),index的维度为(tgt_len-1,b,1)
   #针对dim=-1(vocab_size)维，提取ground truth对应的index的log_softmax值
   torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1)
   ```

8. `torch.squeeze() & torch.unsqueeze()`

   前者在特定维度减少一维（仅当该维=1才有效），后者在特定位置加一维

9. `torch.sum()`

   `dim=0`按照某个维度进行求和，缺省该参数则为所有元素求和。

10. torch.nn.utils.rnn 中pad_packed_sequence和pack_padded_sequence函数，专门用于填充

   `from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence`

   11.1 `pad_packed_sequence`

   [`pad_packed_sequence`]( https://www.cnblogs.com/sbj123456789/p/9834018.html)用于避免pad输入到模型中引起误差，__将一个填充过的长序列压紧__（也就是是说encoder中要用pad补全，但是pad之后才能配合这个函数来实现encoder预计的功能，不pad的话，可能没法并行了都）。

   参数说明:

   - input (Variable) – 变长序列 被填充后(pad后)的 batch
   - lengths (list[int]) – `Variable` 中 每个序列的长度。
   - batch_first (bool, optional) – 如果是`True`，input的形状应该是`B*T*size`。

   返回：PackedSequence对象

   但该函数的使用很方便，直接作为LSTM的参数即可：

   ```python
   enc_hiddens, (last_hidden,last_cell) = self.encoder(pack_padded_sequence(X, source_lengths)) #self.encoder就是LSTM
   #pad_packed_sequence()返回seq_unpacked,lens_unpacked两个部分
   enc_hiddens = pad_packed_sequence(enc_hiddens)[0].permute(1,0,2)
   ```

   11.2 `pack_padded_sequence`

   把压紧的序列再填充回来

   参数说明:

   - sequence (PackedSequence) – 将要被填充的 batch
   - batch_first (bool, optional) – 如果为True，返回的数据的格式为 `B×T×*`

   返回: 一个tuple，包含被填充后的序列，和batch中序列的长度列表

   注意：上文生成的`enc_hiddens`需要经过该函数转换回来，并控制是否`batch_first=True`。但hidden_state(也就是last_hidden,last_cell)并不需要。

11. 判断两个张量是否相等 `torch.equal(tensor1,tensor2)`。`tensor1==tensor2`返回的是与两者同shape的`tensor_bool`

12. `torch.split(tensor,split_size_or_sections,dim=0)`

    该函数主要用于将`tensor`分成块结构，返回的是个`tuple`

    * `dim`控制对哪一维进行分割
    * `split_size_or_sections`若为`int`则为分割__每块含几小块__（不是分割成几部分）；为`list`则代表分割为`len(list)`小块，每块包含小块数与list对应值一致

13. `torch.stack(tensor,dim)`

    作用是将张量“序列”转换为一个完整的张量，list[tgt_len个(b,h)张量]-->(tgt_len,b,h)

    __与cat()函数的区别__：cat在现有维度上加，stack是新加一个维度

14. `torch.bmm`Batch Multiplication,用于两个带batch维的张量间的乘法运算

    (b,n,m)*(b,m,o)=(b,n,p) 

    各自乘各自的，主要用于注意力机制$\left(\mathbf{h}_{t}^{\text {dec }}\right)^{T} $和$\mathbf{W}_{\text {attProj }} \mathbf{h}_{i}^{\text {enc }}$的相乘计算。

    以及得到`a_t`(将attention score概率化)后，和每个$\mathbf{h}_{i}^{\text {enc }}$相乘，得到最终的attention后的结果

15. `torch.expand`函数

    作用为对tensor为1的维度进行复制扩张：若a的shape为 (3,1)，只可a.expand(3,4)；若将维度设为-1则代表维度不改变

    在seq2seq中主要用于：将当前待翻译句子的encodings复制一下，方便基于同一个encoding进行不同的decoding运算，方便beam search。

    `tensor.expand_as()`函数也是类似的用法。

16. `torch.topk`函数

    ```python
    top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
    ```

    前k个最大的，返回相应的最大值与对应的index

    用于`beam search`搜索前k个得分最大的序列

17. 模型的`save`和`load`

    * 模型(参数)保存，可以将参数组合成`dict`，进行存储

      ```python
      params = {
                  'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
                  'vocab': self.vocab,
                  'state_dict': self.state_dict()
              }
      torch.save(params, path)
      ```

    * 模型加载

      ```python
      params = torch.load(model_path, map_location=lambda storage, loc: storage)
      #读的是个字典，剩下的操作和普通的一致
      args = params['args']
      model = NMT(vocab=params['vocab'], **args)# **表示传入的是字典
      model.load_state_dict(params['state_dict'])#load_state_dict的参数必须是state_dict，不可为地址
      ```
    
    * 优化器状态的保存：`optimizer.state_dict()` 

18. 复习`optimizer`的定义方法

    ```python
    opt = torch.optim.SGD(model.parameters(),lr=0.01,**)#普通定义
    opt = torch.optim.Adam([
                           {'params':model.parameters()},
    					 {'params':model.classifier.parameters(),'lr':0.0001}
                           ] ,lr=0.01)#list[dict]
    ```

19. [分类任务损失函数差异](https://www.jb51.net/article/212132.htm)

    多分类使用损失函数`CrossEntropyLoss`，且在模型的输出的最后不需要`softmax`

    多标签任务的损失函数，如果为BCELoss，则需要在模型的输出最后加上sigmoid；如果为BCEWithLogitsLoss，则不需要sigmoid


## NTLK库

1. `from nltk.translate.bleu_score import corpus_bleu` 计算corpus bleu值的

   ```python
   corpus_bleu(reference:list[list[list[str]]],hyp:list[list[str]])#应对多个reference，一个翻译结果的情况
   ```

   这个bleu得分需要乘以100

