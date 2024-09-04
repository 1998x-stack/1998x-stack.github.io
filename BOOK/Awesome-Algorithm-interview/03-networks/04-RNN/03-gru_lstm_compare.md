## GRU相较于LSTM的改动及参数量

### 一、GRU（门控循环单元）的改动

GRU是由Cho等人在2014年提出的一种简化版的LSTM，其设计目的是为了降低计算复杂度，同时保持或提高性能。GRU与LSTM的主要区别如下：

1. **结构简化**：
   - GRU将LSTM中的输入门和遗忘门合并为一个更新门，减少了计算量。
   - GRU没有单独的单元状态，只有一个隐藏状态，因此不需要像LSTM那样维护单元状态。

2. **门控机制**：
   - LSTM有三个门：输入门、遗忘门和输出门，而GRU只有两个门：更新门和重置门。

### 二、GRU的状态更新步骤

1. **更新门 $ z_t $**：
   - 控制当前隐藏状态与前一隐藏状态的混合程度。
   $$
   z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
   $$

2. **重置门 $ r_t $**：
   - 控制当前输入对前一隐藏状态的影响程度。
   $$
   r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
   $$

3. **候选隐藏状态 $ \tilde{h}_t $**：
   - 结合当前输入和前一隐藏状态计算出候选隐藏状态。
   $$
   \tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
   $$

4. **隐藏状态 $ h_t $**：
   - 根据更新门的值，决定当前隐藏状态的更新方式。
   $$
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
   $$

### 三、GRU的参数量

假设输入向量的维度为 $ n $，隐藏状态的维度为 $ m $，则GRU的参数量如下：

1. **更新门参数**：
   - 权重矩阵 $ W_z \in \mathbb{R}^{m \times (m + n)} $ 的参数量为 $ m \times (m + n) $
   - 偏置向量 $ b_z \in \mathbb{R}^m $ 的参数量为 $ m $

2. **重置门参数**：
   - 权重矩阵 $ W_r \in \mathbb{R}^{m \times (m + n)} $ 的参数量为 $ m \times (m + n) $
   - 偏置向量 $ b_r \in \mathbb{R}^m $ 的参数量为 $ m $

3. **候选隐藏状态参数**：
   - 权重矩阵 $ W \in \mathbb{R}^{m \times (m + n)} $ 的参数量为 $ m \times (m + n) $
   - 偏置向量 $ b \in \mathbb{R}^m $ 的参数量为 $ m $

综上所述，GRU的总参数量为：
$$ 3 \times (m \times (m + n) + m) = 3m \times (m + n + 1) $$

### 四、总结

GRU通过简化门控机制，减少了计算复杂度，同时保持了与LSTM相近的性能。在实际应用中，GRU比LSTM更高效，尤其在计算资源有限的情况下。其参数量也相对较少，具体计算如上所示。

### 参考资料

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Cho et al., 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [Comparison of LSTM and GRU on Speech Recognition](https://www.deeplearning.ai/ai-notes/comparison-lstm-gru-rnn/)