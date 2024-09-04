> lstm各个门分别使用什么激活函数？以及为什么？


### LSTM各个门分别使用的激活函数及其原因

长短期记忆网络（LSTM）的核心是其门机制（遗忘门、输入门、输出门），这些门通过不同的激活函数来控制信息流的动态。以下是LSTM中各个门使用的激活函数及其原因：

#### 1. 遗忘门（Forget Gate）

**激活函数**：Sigmoid 函数

**公式**：
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

**原因**：
- **输出范围**：Sigmoid函数的输出范围在0到1之间，这非常适合作为权重，表示每个单元需要保留多少以前的状态。
- **控制信息流**：遗忘门决定了哪些信息需要被保留，哪些需要被遗忘。通过将输出限制在0到1之间，可以精确地控制这些比例。

#### 2. 输入门（Input Gate）

**激活函数**：Sigmoid 函数

**公式**：
$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$

**原因**：
- **输出范围**：同样，Sigmoid函数的输出在0到1之间，表示新信息应该被写入记忆单元的比例。
- **控制信息流**：输入门控制新信息写入记忆单元的比例，输出0表示完全忽略，输出1表示完全接纳。

#### 3. 候选记忆单元（Candidate Cell State）

**激活函数**：Tanh 函数

**公式**：
$$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

**原因**：
- **输出范围**：Tanh函数的输出范围在-1到1之间，适合表示可能的候选记忆值。
- **平衡信息**：Tanh函数提供了正负值的平衡输出，有助于表示和调整信息的流入和流出。

#### 4. 输出门（Output Gate）

**激活函数**：Sigmoid 函数

**公式**：
$$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$

**原因**：
- **输出范围**：Sigmoid函数的输出在0到1之间，控制从记忆单元输出到隐藏状态的比例。
- **控制信息流**：输出门通过Sigmoid函数决定当前记忆单元中哪些部分应该影响到下一个时间步的隐藏状态。

#### 5. 隐藏状态（Hidden State）

**激活函数**：Tanh 函数

**公式**：
$$ h_t = o_t * \tanh(C_t) $$

**原因**：
- **平衡输出**：Tanh函数的输出范围在-1到1之间，使得隐藏状态的值既可以是正数也可以是负数，有助于后续层的学习。
- **非线性变换**：Tanh函数提供了非线性变换，有助于网络表达复杂的模式。

### 参考文献

1. **"Long Short-Term Memory" by Hochreiter and Schmidhuber**：
   - 详细描述了LSTM的结构和工作原理。
   - [论文链接](https://www.bioinf.jku.at/publications/older/2604.pdf)

2. **"Understanding LSTM Networks" by Christopher Olah**：
   - 这篇博客文章通俗易懂地解释了LSTM的工作原理和各个门的作用。
   - [博客链接](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

3. **"Neural Networks and Deep Learning" by Michael Nielsen**：
   - 本书介绍了深度学习的基础知识，包括LSTM在内的多种神经网络结构。
   - [在线书籍链接](http://neuralnetworksanddeeplearning.com/)

