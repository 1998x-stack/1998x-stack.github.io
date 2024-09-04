> 详细展开 word2vec的原理


### Word2Vec的原理

Word2Vec是一种用于学习词向量表示的技术，由Tomas Mikolov及其团队在2013年提出。它通过将词映射到低维连续向量空间中，使得词与词之间的语义关系得以保留。Word2Vec主要有两种模型架构：Skip-gram和CBOW（Continuous Bag of Words）。以下是对这两种模型的详细解释：

#### 1. Skip-gram模型

Skip-gram模型的目标是根据中心词（输入词）预测其上下文词（目标词）。换句话说，对于给定的中心词，模型尝试预测在其窗口范围内出现的所有上下文词。

**模型架构**：
- 输入层：每个输入词表示为一个独热编码向量（one-hot vector）。
- 投影层：通过嵌入矩阵将高维独热向量映射到低维稠密向量（词向量）。
- 输出层：计算每个上下文词的概率分布，通常使用Softmax函数。

**目标函数**：
- 最大化在给定中心词的情况下生成上下文词的概率。
- 使用负采样（Negative Sampling）或分层Softmax（Hierarchical Softmax）来提高计算效率。

**数学表达**：
- 目标函数：$
\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)
$
其中，$T$是词序列的总长度，$c$是窗口大小，$w_t$是中心词，$w_{t+j}$是上下文词。
- 条件概率：$
P(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w \in V} \exp(\mathbf{v}_w \cdot \mathbf{v}_{w_I})}
$
其中，$\mathbf{v}_{w_O}$和$\mathbf{v}_{w_I}$分别是上下文词和中心词的词向量，$V$是词汇表。

#### 2. CBOW模型

CBOW模型的目标是根据上下文词预测中心词。与Skip-gram相反，CBOW通过输入窗口内所有上下文词，预测位于窗口中心的词。

**模型架构**：
- 输入层：窗口内所有上下文词的独热编码向量。
- 投影层：将上下文词向量求平均，得到一个上下文向量。
- 输出层：计算中心词的概率分布，通常使用Softmax函数。

**目标函数**：
- 最大化在给定上下文词的情况下生成中心词的概率。

**数学表达**：
- 目标函数：$
\frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c})
$
其中，$T$是词序列的总长度，$c$是窗口大小。
- 条件概率：与Skip-gram相同，使用Softmax函数计算。

#### 3. 负采样（Negative Sampling）

负采样是一种改进的训练方法，通过简化目标函数来提高训练效率。它通过在训练过程中仅更新少量的负样本，从而加速模型训练。

**负采样的目标函数**：
$$
\log \sigma(\mathbf{v}_{w_O} \cdot \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_I})]
$$
其中，$\sigma(x)$是Sigmoid函数，$k$是负样本数量，$P_n(w)$是负样本的概率分布。

### 实现示例（使用Python的gensim库）

以下是使用gensim库实现Word2Vec的示例代码：

```python
from gensim.models import Word2Vec

# 示例数据
sentences = [
    ['i', 'love', 'machine', 'learning'],
    ['word2vec', 'is', 'a', 'great', 'tool'],
    ['i', 'enjoy', 'learning', 'new', 'things']
]

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # sg=1表示使用Skip-gram模型

# 获取词向量
vector = model.wv['machine']
print("词向量：\n", vector)
```

### 参考文献

1. **Distributed Representations of Words and Phrases and their Compositionality by Tomas Mikolov et al.**：
   - 提供了Word2Vec的详细理论和实现。
   - [论文链接](https://arxiv.org/abs/1310.4546)

2. **Efficient Estimation of Word Representations in Vector Space by Tomas Mikolov et al.**：
   - 介绍了Skip-gram和CBOW模型，以及负采样方法。
   - [论文链接](https://arxiv.org/abs/1301.3781)

3. **Gensim文档**：
   - 提供了使用Gensim库进行Word2Vec训练的实际示例和教程。
   - [Gensim文档](https://radimrehurek.com/gensim/models/word2vec.html)
